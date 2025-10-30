# services/extractors/figure_extractor.py
import subprocess
import json
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import httpx
from dataclasses import dataclass
import io

from app.models.schemas import Figure, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError
from app.services.cloudinary_service import cloudinary_service
# OCR functionality removed for memory optimization

FIGURE_UPLOAD_METHODS = {"pdfplumber", "cv_contour", "chart_axes", "chart_axes_grouped"}
from app.services.ocr.ocr_manager import OCRManager


@dataclass
class FigureCandidate:
    """Intermediate representation for figure candidates"""
    bbox: BoundingBox
    confidence: float
    method: str
    image_path: Optional[str] = None
    caption: Optional[str] = None
    label: Optional[str] = None
    page: int = 1
    ocr_text: Optional[str] = None  # OCR extracted text from the image
    ocr_confidence: Optional[float] = None  # OCR confidence score
    page_width: Optional[float] = None  # Page width for validation
    page_height: Optional[float] = None  # Page height for validation


class FigureExtractor:
    """
    Enhanced multi-method figure extraction using computer vision techniques
    with advanced validation and OCR text extraction for LLM processing
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "figures"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize OCR manager if OCR is enabled
        self.ocr_manager = OCRManager() if settings.use_ocr else None
        if self.ocr_manager and self.ocr_manager.is_any_provider_available():
            logger.info("OCR functionality enabled with OCR.space integration")
        else:
            logger.info("OCR functionality disabled - use_ocr=False or no providers available")
        
        # Caption detection patterns
        self.CAPTION_PATTERNS = [
            r'(?:Figure|Fig\.?)\s*(\d+)[:.]?\s*([^.]*(?:\.[^.]*){0,2})',  # Figure 1: Caption
            r'(?:Figure|Fig\.?)\s*(\d+)\s*[-–—]\s*([^.]*(?:\.[^.]*){0,2})',  # Figure 1 - Caption
            r'(\d+)\.\s*([^.]*(?:\.[^.]*){0,2})',  # 1. Caption (when near figure)
        ]
        
        # Figure validation thresholds - made more strict
        self.MIN_FIGURE_AREA = 15000  # Minimum area in pixels² (increased from 5000)
        self.MAX_FIGURE_AREA = 500000  # Maximum area in pixels²
        self.MIN_ASPECT_RATIO = 0.2
        self.MAX_ASPECT_RATIO = 5.0
        self.MIN_FIGURE_WIDTH = 100  # Minimum width in points
        self.MIN_FIGURE_HEIGHT = 50  # Minimum height in points
        self.MAX_STRIP_ASPECT_RATIO = 20  # Reject very thin strips
    

    
    def _is_ocr_available(self):
        """Check if OCR is available and enabled"""
        return (self.ocr_manager is not None and 
                self.ocr_manager.is_any_provider_available() and 
                settings.use_ocr)
    
    async def _bulk_upload_final_figures(self, figures: List[Figure]) -> None:
        """
        Upload final validated figure images concurrently.
        - Only uploads local paths (not already URLs).
        - Updates figure.image_path in-place to the Cloudinary URL on success.
        """
        if not settings.upload_figures_to_cloud:
            return

        # filter: only eligible methods, path exists, not already a URL
        to_upload = []
        for f in figures:
            if (f.extraction_method in FIGURE_UPLOAD_METHODS and
                isinstance(f.image_path, str) and
                f.image_path and not f.image_path.startswith("http")):
                to_upload.append(f)

        if not to_upload:
            return

        sem = asyncio.Semaphore(settings.upload_concurrency)

        async def _upload_one(fig: Figure):
            path = Path(fig.image_path)
            if not path.exists():
                return
            async with sem:
                try:
                    # mirror method into subfolder to keep things tidy
                    folder = f"{settings.cloudinary_figures_folder}/{fig.extraction_method}"
                    # Upload file to Cloudinary
                    url = await cloudinary_service.upload_file(str(path), folder=folder)
                    if url:
                        fig.image_path = url  # replace local path with cloud URL
                        logger.debug(f"Uploaded figure -> {url}")
                except Exception as e:
                    logger.warning(f"Upload failed for {path.name}: {e}")

        await asyncio.gather(*[_upload_one(f) for f in to_upload])
    
    async def extract(self, pdf_path: Path) -> List[Figure]:
        """
        Run PDFPlumber + enhanced CV (contours + axes) on every file, then dedupe/validate.
        This ensures vector charts & multi-panels (not embedded as images) are found.
        """
        candidates: List[FigureCandidate] = []
        
        # Phase 1: PDFPlumber (embedded images)
        try:
            pdfplumber_candidates = await self._extract_with_pdfplumber(pdf_path)
            candidates.extend(pdfplumber_candidates)
            logger.info(f"PDFPlumber extracted {len(pdfplumber_candidates)} figure candidates")
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            
        # Phase 2: Enhanced CV (ALWAYS) — runs contours + axes/legend + panel merge
        try:
            cv_candidates = await self._extract_with_cv_enhanced(pdf_path)
            candidates.extend(cv_candidates)
            # quick method breakdown for sanity
            by_method = {}
            for c in cv_candidates:
                by_method[c.method] = by_method.get(c.method, 0) + 1
            logger.info(f"Enhanced CV extracted {len(cv_candidates)} candidates: {by_method}")
        except Exception as e:
            logger.error(f"Enhanced CV extraction failed: {e}")

        if not candidates:
            logger.info("No candidates found by any method")
            return []

        # Phase 3: Add captions for CV-only candidates (so validation can use text too)
        try:
            candidates = await self._detect_captions(candidates, pdf_path)
        except Exception as e:
            logger.warning(f"Caption detection phase failed: {e}")
        
        # Phase 4: Dedup + validate + boost
        unique_candidates = self._deduplicate_candidates(candidates)
        validated_candidates = [c for c in unique_candidates if self._validate_candidate(c)]
        validated_candidates = self._boost_chart_confidence(validated_candidates)
        
        # Phase 5: Build final objects
        figures: List[Figure] = []
        for i, candidate in enumerate(validated_candidates):
            try:
                # ensure image exists now
                if not candidate.image_path:
                    candidate.image_path = self._materialize_image_from_pdf(pdf_path, candidate)

                figure = Figure(
                    id=f"figure_{candidate.page}_{i}",
                    label=candidate.label or f"Figure {candidate.page}.{i}",
                    caption=candidate.caption,
                    page=candidate.page,
                    bbox=candidate.bbox,
                    extraction_method=candidate.method,
                    confidence=candidate.confidence,
                    image_path=candidate.image_path,
                    ocr_text=candidate.ocr_text,
                    ocr_confidence=candidate.ocr_confidence
                )
                figures.append(figure)
            except Exception as e:
                logger.warning(f"Failed to create figure from candidate {i}: {e}")
        
        logger.info(
            "Final figures: total=%d (methods=%s)",
            len(figures),
            {f.extraction_method for f in figures}
        )

        # NEW: bulk upload at the end
        try:
            await self._bulk_upload_final_figures(figures)
        except Exception as e:
            logger.warning(f"Bulk upload step failed: {e}")

        return figures
    

    

    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> List[FigureCandidate]:
        """Extract figures using PDFPlumber for high-quality image extraction"""
        candidates = []
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"PDFPlumber opened PDF with {len(pdf.pages)} pages")
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract images from page
                    if hasattr(page, 'images') and page.images:
                        logger.info(f"Page {page_num} has {len(page.images)} images")
                        for img_idx, img_info in enumerate(page.images):
                            try:
                                # Get page dimensions for validation
                                page_width = page.width
                                page_height = page.height
                                
                                # Extract original coordinates
                                x0, top, x1, bottom = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
                                
                                # Check if image is completely or mostly outside page bounds
                                if (x1 <= 0 or x0 >= page_width or 
                                    bottom <= 0 or top >= page_height or
                                    x0 < -50 or x1 < -50 or  # Reject images starting far outside
                                    top < -50 or bottom < -50):
                                    logger.debug(f"Skipping out-of-bounds image {img_idx} on page {page_num}: bounds ({x0:.1f}, {top:.1f}, {x1:.1f}, {bottom:.1f}) vs page ({page_width:.1f}, {page_height:.1f})")
                                    continue
                                
                                # Calculate dimensions before clamping to validate size
                                orig_width = abs(x1 - x0)
                                orig_height = abs(bottom - top)
                                orig_area = orig_width * orig_height
                                
                                # Apply strict size filters before processing
                                if (orig_width < self.MIN_FIGURE_WIDTH or 
                                    orig_height < self.MIN_FIGURE_HEIGHT or
                                    orig_area < self.MIN_FIGURE_AREA):
                                    logger.debug(f"Skipping small image {img_idx} on page {page_num}: size {orig_width:.1f}x{orig_height:.1f} (area: {orig_area:.0f})")
                                    continue
                                
                                # Check for very thin strips (likely artifacts)
                                aspect_ratio = orig_width / orig_height if orig_height > 0 else float('inf')
                                if aspect_ratio > self.MAX_STRIP_ASPECT_RATIO or aspect_ratio < (1/self.MAX_STRIP_ASPECT_RATIO):
                                    logger.debug(f"Skipping strip-like image {img_idx} on page {page_num}: aspect ratio {aspect_ratio:.2f}")
                                    continue
                                
                                # Clamp coordinates to page boundaries (for processing valid images)
                                clamped_x1 = max(0, min(x0, page_width))
                                clamped_y1 = max(0, min(top, page_height))
                                clamped_x2 = max(0, min(x1, page_width))
                                clamped_y2 = max(0, min(bottom, page_height))
                                
                                # Skip if clamping resulted in invalid box
                                if clamped_x1 >= clamped_x2 or clamped_y1 >= clamped_y2:
                                    logger.debug(f"Skipping image {img_idx} on page {page_num}: invalid box after clamping")
                                    continue
                                
                                # Check if clamping significantly reduced the image size
                                clamped_width = clamped_x2 - clamped_x1
                                clamped_height = clamped_y2 - clamped_y1
                                size_retention = (clamped_width * clamped_height) / orig_area
                                
                                if size_retention < 0.7:  # Lost more than 30% of original size
                                    logger.debug(f"Skipping image {img_idx} on page {page_num}: too much size lost in clamping ({size_retention:.2f})")
                                    continue
                                
                                # Create bounding box from clamped coordinates
                                bbox = BoundingBox(
                                    x1=clamped_x1,
                                    y1=clamped_y1,
                                    x2=clamped_x2,
                                    y2=clamped_y2,
                                    page=page_num,
                                    confidence=0.9
                                )
                                
                                # Save image locally using proper PDFPlumber method
                                image_path = None
                                if settings.store_locally:
                                    # Store locally in organized folder structure
                                    pdfplumber_dir = self.output_dir / "pdfplumber"
                                    pdfplumber_dir.mkdir(parents=True, exist_ok=True)
                                    img_path = pdfplumber_dir / f"page{page_num}_img{img_idx}.png"
                                    
                                    # Use proper PDFPlumber image extraction method
                                    # Crop the page to the image region and convert to image
                                    # Use clamped coordinates for cropping to ensure we stay within page bounds
                                    img_bbox = (clamped_x1, clamped_y1, clamped_x2, clamped_y2)
                                    # Expand bbox to include nearby text (titles, captions, legends)
                                    img_bbox = self._expand_bbox_with_nearby_text_pdfplumber(page, img_bbox)
                                    # Add small padding to avoid tight cuts on labels/colorbars
                                    x0, y0, x1, y1 = img_bbox
                                    w, h = x1-x0, y1-y0
                                    pad = max(3, int(0.03 * max(w, h)))  # 3% padding
                                    img_bbox = (max(0, x0-pad), max(0, y0-pad), 
                                               min(page.width, x1+pad), min(page.height, y1+pad))
                                    cropped_region = page.crop(img_bbox)
                                    table_img = cropped_region.to_image(resolution=300)  # Enhanced quality
                                    
                                    # Save as PNG with high quality
                                    table_img.save(str(img_path), format="PNG", optimize=False)  # No compression for quality
                                    
                                    image_path = str(img_path)
                                    logger.debug(f"Stored PDFPlumber image locally: {image_path}")
                                else:
                                    # Upload to Cloudinary
                                    filename = f"pdfplumber_page{page_num}_img{img_idx}"
                                    
                                    # Create image for Cloudinary upload
                                    # Use clamped coordinates for cropping to ensure we stay within page bounds
                                    img_bbox = (clamped_x1, clamped_y1, clamped_x2, clamped_y2)
                                    # Expand bbox to include nearby text (titles, captions, legends)
                                    img_bbox = self._expand_bbox_with_nearby_text_pdfplumber(page, img_bbox)
                                    # Add small padding to avoid tight cuts on labels/colorbars
                                    x0, y0, x1, y1 = img_bbox
                                    w, h = x1-x0, y1-y0
                                    pad = max(3, int(0.03 * max(w, h)))  # 3% padding
                                    img_bbox = (max(0, x0-pad), max(0, y0-pad), 
                                               min(page.width, x1+pad), min(page.height, y1+pad))
                                    cropped_region = page.crop(img_bbox)
                                    table_img = cropped_region.to_image(resolution=300)  # Enhanced quality
                                    
                                    # Convert to bytes for Cloudinary
                                    import io
                                    img_buffer = io.BytesIO()
                                    table_img.save(img_buffer, format="PNG", optimize=False)  # No compression for quality
                                    img_bytes = img_buffer.getvalue()
                                    
                                    image_path = await cloudinary_service.upload_image_from_bytes(
                                        img_bytes, 
                                        filename, 
                                        "scholarai/figures"
                                    )
                                    logger.debug(f"Uploaded PDFPlumber image to Cloudinary: {image_path}")
                                
                                # Extract caption using PDFPlumber text
                                pdfplumber_caption = self._extract_caption_near_image(page, img_info)
                                
                                # Always use OCR to extract text from the figure image itself
                                ocr_text = ""
                                if self._is_ocr_available():
                                    ocr_result = await self._extract_text_from_figure_image(image_path)
                                    if ocr_result:
                                        ocr_text = ocr_result.get('text', '')
                                        logger.debug(f"OCR extracted {len(ocr_text)} characters from figure {page_num}.{img_idx}")
                                
                                # Create candidate with both caption and OCR text
                                candidate = FigureCandidate(
                                    bbox=bbox,
                                    confidence=0.9,
                                    method='pdfplumber',
                                    image_path=image_path,
                                    caption=pdfplumber_caption,
                                    label=f"Figure {page_num}.{img_idx}",
                                    page=page_num,
                                    ocr_text=ocr_text,  # Store OCR extracted text from the figure
                                    ocr_confidence=0.8 if ocr_text else None,
                                    page_width=page_width,  # Add page dimensions for validation
                                    page_height=page_height
                                )
                                candidates.append(candidate)
                                
                            except Exception as e:
                                logger.warning(f"Failed to process image {img_idx} on page {page_num}: {e}")
                    else:
                        # Alternative: Try to extract images using page rendering
                        logger.info(f"Page {page_num} has no images attribute, trying alternative method")
                        # This will be handled by the PyMuPDF fallback
                                
        except Exception as e:
            logger.error(f"PDFPlumber figure extraction failed: {e}")
            # Fallback to PyMuPDF for image extraction
            logger.info("Falling back to PyMuPDF for image extraction")
            candidates.extend(await self._extract_images_with_pymupdf(pdf_path))
        
        return candidates
    
    async def _extract_with_cv_contour_only(self, pdf_path: Path) -> List[FigureCandidate]:
        """Extract figures using only CV contour detection (no charts)"""
        candidates = []
        try:
            # Use PyMuPDF to get page images for CV processing
            import fitz
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Convert page to image for CV processing
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                if pix.n == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Use only contour detection (no chart detection)
                contour_candidates = await self._detect_figures_by_contours(img_array, gray, page_num + 1, page)
                candidates.extend(contour_candidates)
                
                pix = None  # Free memory
            
            doc.close()
            
        except Exception as e:
            logger.error(f"CV contour extraction failed: {e}")
        
        return candidates
    
    async def _extract_images_with_pymupdf(self, pdf_path: Path) -> List[FigureCandidate]:
        """Fallback method to extract images using PyMuPDF if PDFPlumber fails"""
        candidates = []
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # Skip CMYK images
                            # Convert to RGB
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                        # Get image dimensions and position
                        img_rect = page.get_image_bbox(img_info)
                        
                        # Create bounding box
                        bbox = BoundingBox(
                            x1=img_rect.x0,
                            y1=img_rect.y0,
                            x2=img_rect.x1,
                            y2=img_rect.y1,
                            page=page_num + 1,
                            confidence=0.85
                        )
                        
                        # Save image locally
                        image_path = None
                        if settings.store_locally:
                            # Store locally in organized folder structure
                            pymupdf_dir = self.output_dir / "pymupdf"
                            pymupdf_dir.mkdir(parents=True, exist_ok=True)
                            img_path = pymupdf_dir / f"page{page_num + 1}_img{img_idx}.png"
                            
                            # Save the image
                            pix.save(str(img_path))
                            image_path = str(img_path)
                            logger.debug(f"Stored PyMuPDF image locally: {image_path}")
                        else:
                            # Upload to Cloudinary
                            filename = f"pymupdf_page{page_num + 1}_img{img_idx}"
                            # Convert pixmap to bytes for Cloudinary
                            img_bytes = pix.tobytes("png")
                            image_path = await cloudinary_service.upload_image_from_bytes(
                                img_bytes, 
                                filename, 
                                "scholarai/figures"
                            )
                            logger.debug(f"Uploaded PyMuPDF image to Cloudinary: {image_path}")
                        
                        # Create candidate
                        candidate = FigureCandidate(
                            bbox=bbox,
                            confidence=0.85,
                            method='pymupdf',
                            image_path=image_path,
                            caption="",  # PyMuPDF doesn't provide caption context easily
                            label=f"Figure {page_num + 1}.{img_idx}",
                            page=page_num + 1
                        )
                        candidates.append(candidate)
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to process PyMuPDF image {img_idx} on page {page_num + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF fallback image extraction failed: {e}")
        
        return candidates
    
    def _extract_caption_near_image(self, page, img_info: dict) -> str:
        """Extract caption text near the image using PDFPlumber and OCR"""
        try:
            # Look for text near the image using PDFPlumber coordinate system
            # PDFPlumber uses 'top' and 'bottom' for Y coordinates
            img_x0, img_top, img_x1, img_bottom = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
            
            # Search for text below or above the image
            caption_text = ""
            
            # Check below image first (text below has higher Y values in PDFPlumber)
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is below image (within reasonable distance)
                if (word_top > img_bottom and word_top < img_bottom + 100 and
                    abs(word_x0 - img_x0) < 50):
                    caption_text += word['text'] + " "
            
            # If no text below, check above
            if not caption_text.strip():
                for word in page.extract_words():
                    word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                    
                    # Check if word is above image (within reasonable distance)
                    if (word_bottom < img_top and word_bottom > img_top - 100 and
                        abs(word_x0 - img_x0) < 50):
                        caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.warning(f"Caption extraction failed: {e}")
            return ""
    
    async def _extract_text_from_figure_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract text content from the figure image itself using OCR"""
        try:
            if not self._is_ocr_available():
                logger.debug("OCR not available for figure text extraction")
                return None
            
            if not image_path:
                logger.debug("Image path is empty")
                return None
            
            # Smart handling based on storage strategy
            if settings.store_locally:
                # Local storage mode: use local file path
                if Path(image_path).exists():
                    ocr_result = await self.ocr_manager.extract_text(Path(image_path))
                    if not ocr_result:
                        logger.debug("OCR extraction returned no results for local figure")
                        return None
                    
                    extracted_text = ocr_result.get('text', '')
                    if not extracted_text.strip():
                        logger.debug("OCR extracted no text content from local figure")
                        return None
                    
                    logger.debug(f"OCR successfully extracted {len(extracted_text)} characters from local figure")
                    return ocr_result
                else:
                    logger.debug(f"Local image path not found: {image_path}")
                    return None
            else:
                # Cloud-only mode: use Cloudinary URL directly
                if image_path.startswith('http'):
                    # Send Cloudinary URL directly to OCR.space API
                    return await self._extract_text_from_cloudinary_url(image_path)
                else:
                    logger.debug(f"Expected Cloudinary URL but got: {image_path}")
                    return None
            
        except Exception as e:
            logger.warning(f"OCR figure text extraction failed: {e}")
            return None
    
    async def _extract_text_from_cloudinary_url(self, cloudinary_url: str) -> Optional[Dict[str, Any]]:
        """Extract text from Cloudinary image URL directly using OCR.space API"""
        try:
            logger.debug(f"Processing Cloudinary image URL directly for OCR: {cloudinary_url}")
            
            # Use OCR manager to extract text directly from URL
            # The OCR manager will handle the URL-based extraction
            ocr_result = await self.ocr_manager.extract_text_from_url(cloudinary_url)
            
            if not ocr_result:
                logger.debug("OCR extraction returned no results for Cloudinary image URL")
                return None
            
            extracted_text = ocr_result.get('text', '')
            if not extracted_text.strip():
                logger.debug("OCR extracted no text content from Cloudinary image URL")
                return None
            
            logger.debug(f"OCR successfully extracted {len(extracted_text)} characters from Cloudinary image URL")
            return ocr_result
            
        except Exception as e:
            logger.warning(f"Failed to process Cloudinary image URL for OCR: {e}")
            return None
    
    async def _extract_caption_with_ocr(self, image_path: str) -> str:
        """Extract caption from image using OCR when available"""
        try:
            if not self._is_ocr_available():
                logger.debug("OCR not available for caption extraction")
                return ""
            
            if not image_path or not Path(image_path).exists():
                logger.debug(f"Image path not valid for OCR: {image_path}")
                return ""
            
            # Use OCR to extract text from the image
            ocr_result = await self.ocr_manager.extract_text(Path(image_path))
            if not ocr_result:
                logger.debug("OCR extraction returned no results")
                return ""
            
            extracted_text = ocr_result.get('text', '')
            if not extracted_text.strip():
                logger.debug("OCR extracted no text content")
                return ""
            
            # Look for caption patterns in OCR text
            caption_text = self._extract_caption_from_ocr_text(extracted_text)
            
            if caption_text:
                logger.debug(f"OCR extracted caption: {caption_text[:100]}...")
            else:
                logger.debug("No caption pattern found in OCR text")
            
            return caption_text
            
        except Exception as e:
            logger.warning(f"OCR caption extraction failed: {e}")
            return ""
    
    def _extract_caption_from_ocr_text(self, ocr_text: str) -> str:
        """Extract caption from OCR text using pattern matching"""
        try:
            if not ocr_text or not ocr_text.strip():
                return ""
            
            # Look for caption patterns in OCR text
            for pattern in self.CAPTION_PATTERNS:
                matches = re.finditer(pattern, ocr_text, re.IGNORECASE)
                for match in matches:
                    # Extract the caption part (group 2 if available, otherwise full match)
                    if len(match.groups()) >= 2 and match.group(2):
                        caption = match.group(2).strip()
                    else:
                        caption = match.group(0).strip()
                    
                    if caption and len(caption) > 5:  # Minimum caption length
                        return caption
            
            # If no pattern matches, try to extract meaningful text
            # Look for lines that might be captions (not too long, not too short)
            lines = ocr_text.split('\n')
            for line in lines:
                line = line.strip()
                if (len(line) > 10 and len(line) < 200 and  # Reasonable caption length
                    not line.isdigit() and  # Not just numbers
                    not line.startswith('http') and  # Not URLs
                    any(char.isalpha() for char in line)):  # Contains letters
                    return line
            
            return ""
            
        except Exception as e:
            logger.warning(f"Caption pattern extraction from OCR text failed: {e}")
            return ""
    
    async def _extract_with_cv(self, pdf_path: Path) -> List[FigureCandidate]:
        """
        Legacy CV extraction method - now calls enhanced version
        """
        return await self._extract_with_cv_enhanced(pdf_path)
    
    def _is_likely_figure(self, img: np.ndarray) -> bool:
        """
        Legacy method - now uses enhanced validation
        """
        return self._is_likely_figure_content(img)
    
    def _deduplicate_figures(self, 
                           existing: List[Figure], 
                           new: List[Figure],
                           iou_threshold: float = 0.5) -> List[Figure]:
        """
        Legacy deduplication method - now uses enhanced version
        """
        # Convert to candidates for enhanced deduplication
        existing_candidates = []
        new_candidates = []
        
        # Convert existing figures to candidates
        for fig in existing:
            candidate = FigureCandidate(
                bbox=fig.bbox,
                confidence=fig.confidence or 0.5,
                method=fig.extraction_method or 'unknown',
                image_path=fig.image_path,
                caption=fig.caption,
                label=fig.label,
                page=fig.page
            )
            existing_candidates.append(candidate)
        
        # Convert new figures to candidates
        for fig in new:
            candidate = FigureCandidate(
                bbox=fig.bbox,
                confidence=fig.confidence or 0.5,
                method=fig.extraction_method or 'unknown',
                image_path=fig.image_path,
                caption=fig.caption,
                label=fig.label,
                page=fig.page
            )
            new_candidates.append(candidate)
        
        # Use enhanced deduplication
        unique_candidates = self._remove_duplicate_candidates(new_candidates, iou_threshold)
        
        # Convert back to figures
        unique_figures = []
        for candidate in unique_candidates:
            figure = Figure(
                label=candidate.label,
                caption=candidate.caption,
                page=candidate.page,
                bbox=candidate.bbox,
                image_path=candidate.image_path,
                type=self._determine_figure_type(candidate),
                extraction_method=candidate.method,
                confidence=candidate.confidence,
                ocr_text=candidate.ocr_text,
                ocr_confidence=candidate.ocr_confidence
            )
            unique_figures.append(figure)
        
        return unique_figures
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        # Calculate intersection
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


    
    def _is_likely_figure_content(self, img: np.ndarray) -> bool:
        """Enhanced heuristic to determine if image contains figure content"""
        if img is None or img.size == 0:
            return False
        
        height, width = img.shape[:2]
        
        # Apply strict size validation
        area = width * height
        if (area < self.MIN_FIGURE_AREA or area > self.MAX_FIGURE_AREA or
            width < self.MIN_FIGURE_WIDTH or height < self.MIN_FIGURE_HEIGHT):
            return False
        
        # Aspect ratio validation with strip detection
        aspect_ratio = width / height
        if (aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO or
            aspect_ratio > self.MAX_STRIP_ASPECT_RATIO or aspect_ratio < (1/self.MAX_STRIP_ASPECT_RATIO)):
            return False
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Content analysis
        score = 0
        
        # 1. Edge density (figures typically have structured content)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if 0.02 <= edge_ratio <= 0.3:  # Reasonable edge density
            score += 1
        
        # 2. Variance (figures have variation, not uniform backgrounds)
        variance = np.var(gray)
        if variance > 200:  # Sufficient variation
            score += 1
        
        # 3. Color diversity (figures often have multiple colors)
        if len(img.shape) == 3:
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors > 50:  # Diverse color palette
                score += 1
        
        # 4. Text content analysis (figures shouldn't be mostly text)
        text_likelihood = self._estimate_text_content(gray)
        if text_likelihood < 0.7:  # Not predominantly text
            score += 1
        
        # 5. Geometric shapes detection
        if self._detect_geometric_patterns(gray):
            score += 1
        
        # Require at least 3 positive indicators
        return score >= 3
    
    def _estimate_text_content(self, gray_img: np.ndarray) -> float:
        """Estimate likelihood that image contains primarily text"""
        # Look for text-like patterns
        # Horizontal lines (text baselines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Vertical spacing patterns typical of text
        vertical_projection = np.sum(horizontal_lines, axis=1)
        
        # Count peaks (potential text lines)
        peaks = []
        for i in range(1, len(vertical_projection) - 1):
            if (vertical_projection[i] > vertical_projection[i-1] and 
                vertical_projection[i] > vertical_projection[i+1] and
                vertical_projection[i] > np.mean(vertical_projection)):
                peaks.append(i)
        
        # Regular spacing suggests text
        if len(peaks) >= 3:
            spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            spacing_variance = np.var(spacings) if spacings else float('inf')
            
            # Low variance in line spacing suggests text
            if spacing_variance < 50:  # Relatively consistent line spacing
                return 0.8
        
        return 0.3  # Low text likelihood
    
    def _detect_geometric_patterns(self, gray_img: np.ndarray) -> bool:
        """Detect geometric patterns common in figures"""
        # Detect lines
        edges = cv2.Canny(gray_img, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None and len(lines) > 5:
            return True
        
        # Detect circles
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None and len(circles[0]) > 0:
            return True
        
        # Detect rectangles/contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = 0
        for contour in contours:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Count rectangular shapes
            if len(approx) == 4:
                rectangles += 1
        
        return rectangles >= 3
    
    async def _extract_with_cv_enhanced(self, pdf_path: Path) -> List[FigureCandidate]:
        """Enhanced computer vision detection with better filtering"""
        candidates = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Render page to high-resolution image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale figure detection
            candidates_page = await self._multi_scale_figure_detection(
                img, gray, page_num, page
            )
            candidates.extend(candidates_page)
        
        doc.close()
        return candidates
    
    async def _multi_scale_figure_detection(self, img: np.ndarray, gray: np.ndarray, 
                                          page_num: int, page) -> List[FigureCandidate]:
        """Multi-scale approach for figure detection"""
        candidates = []
        
        # 1) Contours (good for photos/heatmaps)
        contour_candidates = await self._detect_figures_by_contours(img, gray, page_num, page)
        candidates.extend(contour_candidates)

        # 2) Axes/legend-based chart detection (line plots, multi-panels)
        chart_candidates = self._detect_charts_with_axes(img, gray, page_num, page)
        candidates.extend(chart_candidates)

        # 3) Merge aligned subpanels into one figure (A/B/C/D layouts)
        candidates = self._merge_panels_by_alignment(candidates, page)
        
        return candidates
    
    async def _detect_figures_by_contours(self, img: np.ndarray, gray: np.ndarray, 
                                  page_num: int, page) -> List[FigureCandidate]:
        """Detect figures using contour analysis with clustering and content snapping"""
        candidates = []
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = img.shape[:2]
        page_area = W * H

        raw_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ar = w / h if h > 0 else 999

            if (area > self.MIN_FIGURE_AREA and
                area < min(self.MAX_FIGURE_AREA, 0.6 * page_area) and
                self.MIN_ASPECT_RATIO < ar < self.MAX_ASPECT_RATIO and
                1/self.MAX_STRIP_ASPECT_RATIO <= ar <= self.MAX_STRIP_ASPECT_RATIO):
                raw_boxes.append((x, y, w, h))

        # Merge nearby fragments
        merged = self._cluster_boxes(raw_boxes, gap=40)

        # Snap merged boxes to content to cover full figure
        snapped = [self._snap_to_content(gray, x, y, w, h) for (x, y, w, h) in merged]

        # Create candidates (no file I/O yet)
        scale = page.rect.width / W
        for i, (x, y, w, h) in enumerate(snapped):
            roi = img[y:y+h, x:x+w]
            if not self._validate_figure_region(roi):
                continue

            cand = FigureCandidate(
                bbox=BoundingBox(
                    x1=x*scale, y1=y*scale, x2=(x+w)*scale, y2=(y+h)*scale,
                    page=page_num, confidence=0.78
                ),
                confidence=0.78,
                method='cv_contour',
                image_path=None,  # defer saving
                label=f"Figure {page_num}.contour{i}",
                page=page_num,
                page_width=page.rect.width,
                page_height=page.rect.height
            )
            candidates.append(cand)

        return candidates
    
    def _rects_intersect(self, a, b):
        """Check if two rectangles intersect"""
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)

    def _cluster_boxes(self, boxes, gap=35):
        """
        Cluster overlapping/nearby boxes and return merged (x,y,w,h) boxes.
        Works in raster (page image) space.
        """
        merged = []
        for b in sorted(boxes, key=lambda r: r[2]*r[3], reverse=True):
            x,y,w,h = b
            merged_flag = False
            for i,(X,Y,W,H) in enumerate(merged):
                # near or overlapping?
                if not (x > X+W+gap or X > x+w+gap or y > Y+H+gap or Y > y+h+gap):
                    nx, ny = min(x,X), min(y,Y)
                    nx2, ny2 = max(x+w, X+W), max(y+h, Y+H)
                    merged[i] = (nx, ny, nx2-nx, ny2-ny)
                    merged_flag = True
                    break
            if not merged_flag:
                merged.append(b)
        return merged

    def _snap_to_content(self, gray, x, y, w, h):
        """
        Grow a box to cover the whole figure by snapping to non-white content
        (and small gaps) using local threshold + morphology.
        """
        H, W = gray.shape[:2]
        pad = int(0.25 * max(w, h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return (x, y, w, h)

        # binary content mask (invert = content is white->0, ink->1)
        # Otsu works well across mixed backgrounds
        _, binv = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # close small gaps and join nearby pieces
        binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)), iterations=2)
        binv = cv2.dilate(binv, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=1)

        cnts, _ = cv2.findContours(binv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return (x, y, w, h)

        # keep only contours that intersect the original (shifted) box
        keep = []
        orig = (x - x0, y - y0, w, h)
        for c in cnts:
            rx, ry, rw, rh = cv2.boundingRect(c)
            if self._rects_intersect((rx, ry, rw, rh), orig):
                keep.append((rx, ry, rw, rh))

        if not keep:
            return (x, y, w, h)

        nx = min(k[0] for k in keep)
        ny = min(k[1] for k in keep)
        nx2 = max(k[0]+k[2] for k in keep)
        ny2 = max(k[1]+k[3] for k in keep)
        nx += x0; ny += y0
        nw = nx2 + x0 - nx
        nh = ny2 + y0 - ny

        # clamp to page
        nx = max(0, nx); ny = max(0, ny)
        nw = min(W - nx, nw); nh = min(H - ny, nh)
        return (nx, ny, nw, nh)

    def _overlap_ratio(self, a0, a1, b0, b1):
        """Calculate overlap ratio between two intervals"""
        inter = max(0, min(a1, b1) - max(a0, b0))
        return inter / max(1e-5, (a1 - a0))

    def _expand_bbox_with_nearby_text_pdfplumber(self, page, img_bbox):
        """Expand bbox to include nearby text blocks (titles, captions, legends)"""
        x0, y0, x1, y1 = img_bbox
        w, h = x1-x0, y1-y0
        pad = max(6, int(0.04 * max(w, h)))
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(page.width, x1 + pad), min(page.height, y1 + pad)

        words = page.extract_words(extra_attrs=["size"])
        # coalesce words into rough blocks (same line)
        lines = {}
        for wobj in words:
            key = int(wobj["top"] // max(6, wobj["height"]))
            lines.setdefault(key, []).append(wobj)

        blocks = []
        for _, group in lines.items():
            xs0 = min(g["x0"] for g in group); xs1 = max(g["x1"] for g in group)
            ys0 = min(g["top"] for g in group); ys1 = max(g["bottom"] for g in group)
            text = " ".join(g["text"] for g in sorted(group, key=lambda z: z["x0"]))
            blocks.append((xs0, ys0, xs1, ys1, text))

        # heuristics: title above, caption below, legend right
        for bx0, by0, bx1, by1, text in blocks:
            v_overlap = self._overlap_ratio(by0, by1, y0, y1)
            h_overlap = self._overlap_ratio(bx0, bx1, x0, x1)

            # Title: above, good horizontal overlap
            if by1 <= y0 and (y0 - by1) < 0.30*h and h_overlap > 0.55:
                X0, Y0, X1, Y1 = min(X0, bx0), min(Y0, by0), max(X1, bx1), max(Y1, by1)

            # Caption: below, good horizontal overlap (also catches "Fig. …")
            if by0 >= y1 and (by0 - y1) < 0.40*h and h_overlap > 0.55:
                X0, Y0, X1, Y1 = min(X0, bx0), min(Y0, by0), max(X1, bx1), max(Y1, by1)

            # Legend: to the right, decent vertical overlap
            if bx0 >= x1 and (bx0 - x1) < 0.60*w and v_overlap > 0.35:
                X0, Y0, X1, Y1 = min(X0, bx0), min(Y0, by0), max(X1, bx1), max(Y1, by1)

        return (max(0, X0), max(0, Y0), min(page.width, X1), min(page.height, Y1))

    def _non_orthogonal_edge_ratio(self, gray):
        """Calculate ratio of non-orthogonal edges (charts have more diagonal/curved lines than text)"""
        edges = cv2.Canny(gray, 60, 160)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ang = np.abs(np.arctan2(gy, gx))  # 0..pi
        mask = edges > 0
        if not np.any(mask): return 0.0
        # count angles that are NOT near 0 or 90 deg (text is mostly orthogonal)
        non_ortho = ((ang[mask] > np.deg2rad(15)) & (ang[mask] < np.deg2rad(75))).sum()
        return non_ortho / float(mask.sum())

    def _is_mostly_text_region(self, gray):
        """Detect if region is mostly text (paragraphs) vs. chart content"""
        # Otsu + CC stats: paragraphs = many tiny components + strong orthogonal edge bias
        _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num, _, stats, _ = cv2.connectedComponentsWithStats(binv)
        if num <= 1: return False
        areas = stats[1:, cv2.CC_STAT_AREA]
        small = (areas < 120).sum() / max(1, len(areas))
        # orthogonal edge ratio
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
        ortho = 0
        if lines is not None:
            for r,theta in lines[:,0]:
                a = abs(theta % (np.pi/2))
                if (a < np.deg2rad(10)) or (abs(a - np.pi/2) < np.deg2rad(10)):
                    ortho += 1
        ortho_ratio = ortho / max(1, len(lines[:,0])) if lines is not None else 0
        return (small > 0.65) and (ortho_ratio > 0.75)

    def _materialize_image_from_pdf(self, pdf_path: Path, cand: "FigureCandidate") -> Optional[str]:
        """
        Create/save the image for a validated candidate.
        Uses PDF coords (points) to render a high-res crop.
        """
        try:
            import fitz
            page_index = cand.page - 1
            rect = fitz.Rect(cand.bbox.x1, cand.bbox.y1, cand.bbox.x2, cand.bbox.y2)
            doc = fitz.open(str(pdf_path))
            page = doc[page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)

            if settings.store_locally:
                out_dir = self.output_dir / cand.method
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"page{cand.page}_fig{hash((cand.bbox.x1, cand.bbox.y1, cand.bbox.x2, cand.bbox.y2)) & 0xffff}.png"
                pix.save(str(path))
                doc.close()
                return str(path)
            else:
                img_bytes = pix.tobytes("png")
                doc.close()
                return asyncio.get_event_loop().run_until_complete(
                    cloudinary_service.upload_image_from_bytes(img_bytes, f"page{cand.page}_fig", "scholarai/figures")
                )
        except Exception as e:
            logger.warning(f"Failed to materialize image: {e}")
            return None
    
    def _detect_charts_with_axes(self, img: np.ndarray, gray: np.ndarray, 
                                page_num: int, page) -> List[FigureCandidate]:
        """Detect charts by finding rectangular plotting areas with axes and legends"""
        H, W = gray.shape[:2]
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))

        # Find long near-horizontal & near-vertical lines (axes & borders)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min(W,H)//5, maxLineGap=10)
        rects = []
        if lines is not None:
            # cluster lines by angle; build rectangles from perpendicular pairs
            horiz, vert = [], []
            for x1,y1,x2,y2 in lines[:,0]:
                ang = abs(np.arctan2(y2-y1, x2-x1))
                if ang < np.deg2rad(10): horiz.append((x1,y1,x2,y2))
                elif abs(ang - np.pi/2) < np.deg2rad(10): vert.append((x1,y1,x2,y2))

            # brute pair rectangles from line extremes
            if horiz and vert:
                xs = [x for x,_,x2,_ in vert] + [x2 for x,_,x2,_ in vert]
                ys = [y for _,y,_,y2 in horiz] + [y2 for _,y,_,y2 in horiz]
                if xs and ys:
                    # coarse grid cells (captures A/B/C/D panels)
                    xs = sorted(xs); ys = sorted(ys)
                    for i in range(0, len(xs)-1):
                        for j in range(0, len(ys)-1):
                            x1, x2 = xs[i], xs[i+1]
                            y1, y2 = ys[j], ys[j+1]
                            w, h = x2-x1, y2-y1
                            if w>80 and h>60:   # min size for a panel
                                rects.append((x1,y1,w,h))

        # Validate rectangles as "plot areas"
        chart_boxes = []
        for (x,y,w,h) in rects:
            roi = img[y:y+h, x:x+w]
            if roi.size == 0: 
                continue
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            border = edges[y:y+h, x:x+w]

            score = 0
            # ticks: many short collinear segments along bottom/left edges
            tick_density = (np.sum(border[:5,:]>0) + np.sum(border[-5:,:]>0) +
                            np.sum(border[:,:5]>0) + np.sum(border[:,-5:]>0)) / (2*w + 2*h + 1)
            if tick_density > 0.05: score += 1

            # moderate edge density inside plot
            ed = np.count_nonzero(border)/border.size
            if 0.02 <= ed <= 0.35: score += 1

            # NEW: "not a paragraph"
            if not self._is_mostly_text_region(g): score += 1

            # NEW: there should be a non-orthogonal signal (lines/curves of data)
            if self._non_orthogonal_edge_ratio(g) > 0.12: score += 1

            # hints of legend: small colored lines + adjacent text to the right or inside
            legend_ok = self._heuristic_legend_hint(img, x, y, w, h)
            if legend_ok: score += 1

            if score >= 4:
                # expand a bit to include labels/colorbars
                pad = int(0.08*max(w,h))
                bx, by = max(0,x-pad), max(0,y-pad)
                bw, bh = min(W-bx, w+2*pad), min(H-by, h+2*pad)
                chart_boxes.append((bx,by,bw,bh))

        # Merge overlapping/nearby plot boxes (panels)
        merged = self._merge_boxes(chart_boxes, gap=25)

        cands = []
        scale = page.rect.width / W
        for i,(x,y,w,h) in enumerate(merged):
            # Grow to full figure boundary
            x,y,w,h = self._snap_to_content(gray, x, y, w, h)

            cands.append(FigureCandidate(
                        bbox=BoundingBox(
                    x1=x*scale, y1=y*scale, x2=(x+w)*scale, y2=(y+h)*scale,
                    page=page_num, confidence=0.85
                ),
                confidence=0.85,
                method='chart_axes',
                image_path=None,           # defer saving
                label=f"Figure {page_num}.chart{i}",
                page=page_num,
                page_width=page.rect.width,
                page_height=page.rect.height
            ))
        return cands
    
    def _merge_boxes(self, boxes, gap=20):
        """Simple NMS-like merge of overlapping/nearby rectangles"""
        out = []
        for b in sorted(boxes, key=lambda r:r[2]*r[3], reverse=True):
            x,y,w,h = b
            merged = False
            for k,(X,Y,W,H) in enumerate(out):
                if not (x> X+W+gap or X> x+w+gap or y> Y+H+gap or Y> y+h+gap):
                    # merge
                    nx = min(x,X); ny=min(y,Y)
                    nx_max = max(x+w, X+W); ny_max = max(y+h, Y+H)
                    out[k] = (nx,ny,nx_max-nx,ny_max-ny)
                    merged = True
                    break
            if not merged: out.append(b)
        return out

    def _heuristic_legend_hint(self, img, x, y, w, h):
        """Look slightly right/top of the plot for short colored line segments + texty region"""
        H, W = img.shape[:2]
        rx1 = min(W-1, x+w+5); ry1 = max(0, y-5)
        rx2 = min(W-1, x+w+int(0.35*w)); ry2 = min(H-1, y+h+int(0.35*h))
        if rx2<=rx1 or ry2<=ry1: return False
        region = img[ry1:ry2, rx1:rx2]
        if region.size == 0: return False
        # lots of small edges but low paragraph density ~ legend bullets
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        ed = np.count_nonzero(cv2.Canny(gray, 60, 160))/gray.size
        text_like = self._estimate_text_content(gray)
        return ed>0.03 and text_like<0.7

    
    def _merge_panels_by_alignment(self, candidates, page, align_tol=12):
        """Merge aligned subpanels into single figures (A/B/C/D layouts)"""
        # group by row/col alignment in page space
        merged = []
        used = set()
        
        # Separate contour and chart candidates for IoU checking
        contour_candidates = [c for c in candidates if c.method == 'cv_contour']
        chart_candidates = [c for c in candidates if c.method == 'chart_axes']
        
        for i,a in enumerate(candidates):
            if i in used: continue
            group = [a]
            for j,b in enumerate(candidates):
                if j in used or i==j: continue
                if abs(a.bbox.y1 - b.bbox.y1)<align_tol or abs(a.bbox.x1 - b.bbox.x1)<align_tol:
                    group.append(b); used.add(j)
            if len(group)>=2:
                x1=min(g.bbox.x1 for g in group); y1=min(g.bbox.y1 for g in group)
                x2=max(g.bbox.x2 for g in group); y2=max(g.bbox.y2 for g in group)
                
                # Check if grouped region has IoU with any contour candidate (≥0.25)
                has_contour_overlap = False
                for contour_cand in contour_candidates:
                    iou = self._calculate_bbox_iou(
                        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page=a.page),
                        contour_cand.bbox
                    )
                    if iou >= 0.25:
                        has_contour_overlap = True
                        break
                
                # Only keep grouped regions that overlap with contour candidates
                if has_contour_overlap:
                    merged.append(FigureCandidate(
                        bbox=BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            page=a.page,
                            confidence=max(g.bbox.confidence for g in group),
                        ),
                        confidence=max(g.confidence for g in group),
                        method='chart_axes_grouped',
                        image_path=None,
                        label=f"Figure {a.page}.grid",
                        page=a.page,
                        page_width=a.page_width,
                        page_height=a.page_height,
                    ))
                    used.update({i})
                else:
                    # If no contour overlap, add individual candidates back
                    for g in group:
                        if g not in merged:
                            merged.append(g)
            else:
                merged.append(a)
        return merged
    
    def _boost_chart_confidence(self, candidates: List[FigureCandidate]) -> List[FigureCandidate]:
        """Boost confidence for figures with chart-specific text tokens"""
        CHART_TOKENS = {
            "TP fraction", "Number of pairs", "iteration", "Random", "DCA", "Hung.", 
            "Orthology", "Distance", "Mirrortree", "amino-acid", "DCA score", 
            "Rank of", "Average", "training set", "species", "mutation", "chain"
        }
        
        for candidate in candidates:
            # Check caption text
            if candidate.caption:
                caption_lower = candidate.caption.lower()
                if any(token.lower() in caption_lower for token in CHART_TOKENS):
                    candidate.confidence += 0.05
                    logger.debug(f"Boosted confidence for chart-like caption: {candidate.label}")
            
            # Check OCR text
            if candidate.ocr_text:
                ocr_lower = candidate.ocr_text.lower()
                if any(token.lower() in ocr_lower for token in CHART_TOKENS):
                    candidate.confidence += 0.05
                    logger.debug(f"Boosted confidence for chart-like OCR text: {candidate.label}")
        
        return candidates
    
    def _expand_figure_boundary(self, img: np.ndarray, gray: np.ndarray, 
                               x: int, y: int, w: int, h: int, page_area: int) -> Optional[tuple]:
        """Intelligently expand figure boundary to capture captions, labels, and sub-components"""
        height, width = img.shape[:2]
        
        # Initial expansion factors
        expand_x = int(w * 0.2)  # 20% horizontal expansion
        expand_y = int(h * 0.3)  # 30% vertical expansion (more for captions)
        
        # Calculate new boundaries
        new_x = max(0, x - expand_x)
        new_y = max(0, y - expand_y)
        new_w = min(width - new_x, w + 2 * expand_x)
        new_h = min(height - new_y, h + 2 * expand_y)
        
        # Look for text content around the figure
        text_regions = self._detect_text_regions_around_figure(gray, x, y, w, h)
        
        # Expand to include detected text regions
        for tx, ty, tw, th in text_regions:
                            # Check if text region is close to the figure
            if self._is_text_region_near_figure(tx, ty, tw, th, x, y, w, h):
                new_x = min(new_x, tx)
                new_y = min(new_y, ty)
                new_w = max(new_w, tx + tw - new_x)
                new_h = max(new_h, ty + th - new_y)
        
        # Look for connected components that might be part of the figure
        connected_regions = self._find_connected_figure_components(gray, x, y, w, h)
        
        # Expand to include connected components
        for cx, cy, cw, ch in connected_regions:
            new_x = min(new_x, cx)
            new_y = min(new_y, cy)
            new_w = max(new_w, cx + cw - new_x)
            new_h = max(new_h, cy + ch - new_y)
        
        # Ensure boundaries don't exceed page limits
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(width - new_x, new_w)
        new_h = min(height - new_y, new_h)
        
        # Validate expanded region
        new_area = new_w * new_h
        if new_area > page_area * 0.6:  # Don't expand too much
            return None
        
        # Check if expansion is reasonable
        expansion_ratio = new_area / (w * h)
        if expansion_ratio > 3.0:  # Don't expand more than 3x
            return None
        
        return (new_x, new_y, new_w, new_h)
    
    def _detect_text_regions_around_figure(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> List[tuple]:
        """Detect text regions around the figure that might be captions or labels"""
        text_regions = []
        
        # Define search area around the figure
        search_margin = max(w, h) // 2
        search_x = max(0, x - search_margin)
        search_y = max(0, y - search_margin)
        search_w = min(gray.shape[1] - search_x, w + 2 * search_margin)
        search_h = min(gray.shape[0] - search_y, h + 2 * search_margin)
        
        # Extract search region
        search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Use morphological operations to detect text-like patterns
        # Horizontal lines (text baselines)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(search_region, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours in horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            tx, ty, tw, th = cv2.boundingRect(contour)
            if tw > 50 and th > 5:  # Minimum text size
                # Convert back to page coordinates
                tx += search_x
                ty += search_y
                text_regions.append((tx, ty, tw, th))
        
        return text_regions
    
    def _is_text_region_near_figure(self, tx: int, ty: int, tw: int, th: int, 
                                   fx: int, fy: int, fw: int, fh: int) -> bool:
        """Check if text region is close to the figure"""
        # Calculate centers
        text_center_x = tx + tw // 2
        text_center_y = ty + th // 2
        figure_center_x = fx + fw // 2
        figure_center_y = fy + fh // 2
        
        # Calculate distance
        distance = ((text_center_x - figure_center_x) ** 2 + 
                   (text_center_y - figure_center_y) ** 2) ** 0.5
        
        # Check if text is within reasonable distance
        max_distance = max(fw, fh) * 1.5
        return distance < max_distance
    
    def _find_connected_figure_components(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> List[tuple]:
        """Find connected components that might be part of the figure"""
        connected_regions = []
        
        # Define search area
        search_margin = max(w, h) // 3
        search_x = max(0, x - search_margin)
        search_y = max(0, y - search_margin)
        search_w = min(gray.shape[1] - search_x, w + 2 * search_margin)
        search_h = min(gray.shape[0] - search_y, h + 2 * search_margin)
        
        # Extract search region
        search_region = gray[search_y:search_y+search_h, search_x:search_x+search_w]
        
        # Threshold to create binary image
        _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        
        for i in range(1, num_labels):  # Skip background
            cx, cy, cw, ch, area = stats[i]
            
            # Check if component is reasonably sized and close to figure
            if (area > 100 and area < w * h * 2 and  # Not too small or too large
                self._is_text_region_near_figure(cx, cy, cw, ch, x-search_x, y-search_y, w, h)):
                
                # Convert back to page coordinates
                cx += search_x
                cy += search_y
                connected_regions.append((cx, cy, cw, ch))
        
        return connected_regions
    

    

    
    def _group_chart_regions(self, contours: List, img_shape: tuple) -> List[tuple]:
        """Group nearby chart intersections to avoid fragmentation"""
        if not contours:
            return []
        
        # Get bounding boxes for all contours
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
        
        # Sort by area (largest first) to prioritize bigger regions
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        
        # Group overlapping or nearby regions
        merged_regions = []
        used = set()
        
        for i, (x1, y1, w1, h1) in enumerate(regions):
            if i in used:
                continue
                
            # Start with current region
            merged_x, merged_y = x1, y1
            merged_w, merged_h = w1, h1
            used.add(i)
            
            # Look for nearby regions to merge
            for j, (x2, y2, w2, h2) in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if regions are close enough to merge
                distance_threshold = 100  # pixels
                
                # Calculate center points
                cx1, cy1 = x1 + w1//2, y1 + h1//2
                cx2, cy2 = x2 + w2//2, y2 + h2//2
                
                # Calculate distance between centers
                distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                
                # Check for overlap or proximity
                overlap = (x1 < x2 + w2 and x2 < x1 + w1 and 
                          y1 < y2 + h2 and y2 < y1 + h1)
                
                if overlap or distance < distance_threshold:
                    # Merge regions
                    merged_x = min(merged_x, x2)
                    merged_y = min(merged_y, y2)
                    merged_w = max(merged_x + merged_w, x2 + w2) - merged_x
                    merged_h = max(merged_y + merged_h, y2 + h2) - merged_y
                    used.add(j)
            
            # Add some padding to capture full chart
            padding = 50
            merged_x = max(0, merged_x - padding)
            merged_y = max(0, merged_y - padding)
            merged_w = min(img_shape[1] - merged_x, merged_w + 2 * padding)
            merged_h = min(img_shape[0] - merged_y, merged_h + 2 * padding)
            
            merged_regions.append((merged_x, merged_y, merged_w, merged_h))
        
        return merged_regions
    
    def _validate_chart_region(self, roi: np.ndarray) -> bool:
        """Validate if region contains chart-like content"""
        if roi.size == 0:
            return False
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Calculate various features to distinguish charts from text
        score = 0
        
        # 1. Edge density analysis
        edges = cv2.Canny(gray_roi, 30, 100)
        edge_density = np.count_nonzero(edges) / edges.size
        
        # Charts typically have moderate edge density (not too sparse, not too dense)
        if 0.03 <= edge_density <= 0.3:
            score += 1
        
        # 2. Line detection (charts have more lines than text)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        if lines is not None and len(lines) > 5:
            score += 1
        
        # 3. Text content analysis (charts should have less text than figures)
        text_likelihood = self._estimate_text_content(gray_roi)
        if text_likelihood < 0.6:  # Lower threshold for charts
            score += 1
        
        # 4. Variance analysis (charts have more visual variation than text)
        variance = np.var(gray_roi)
        if variance > 150:  # Charts have more variation
            score += 1
        
        # 5. Aspect ratio check (charts are often wider than tall)
        h, w = gray_roi.shape
        aspect_ratio = w / h
        if 0.5 <= aspect_ratio <= 3.0:  # Reasonable chart aspect ratio
            score += 1
        
        # 6. Color diversity (charts often have multiple colors)
        if len(roi.shape) == 3:
            unique_colors = len(np.unique(roi.reshape(-1, roi.shape[2]), axis=0))
            if unique_colors > 30:  # Charts have more colors
                score += 1
        
        # Require at least 4 positive indicators for a chart
        return score >= 4
    
    def _detect_diagram_patterns(self, img: np.ndarray, gray: np.ndarray, 
                               page_num: int, page) -> List[FigureCandidate]:
        """Detect diagram patterns with geometric shapes"""
        # This method can be expanded based on specific diagram types
        # For now, return empty list
        _ = img, gray, page_num, page  # Suppress unused parameter warnings
        return []
    

    

    
    def _validate_figure_region(self, roi: np.ndarray) -> bool:
        """General validation for figure regions with improved tolerance for expanded boundaries"""
        if roi.size == 0:
            return False
        
        # For expanded boundaries, be more lenient with validation
        # since they may include captions and labels
        return self._is_likely_figure_content_expanded(roi)
    
    def _is_likely_figure_content_expanded(self, img: np.ndarray) -> bool:
        """Enhanced validation for expanded figure regions that may include captions"""
        if img is None or img.size == 0:
            return False
        
        height, width = img.shape[:2]
        
        # More lenient size validation for expanded regions
        area = width * height
        if area < self.MIN_FIGURE_AREA * 0.5 or area > self.MAX_FIGURE_AREA * 1.5:
            return False
        
        # More flexible aspect ratio for expanded regions
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Much more flexible
            return False
        
        # Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Content analysis with lower thresholds
        score = 0
        
        # 1. Edge density (more lenient for expanded regions)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if 0.01 <= edge_ratio <= 0.4:  # More lenient range
            score += 1
        
        # 2. Variance (figures have variation, not uniform backgrounds)
        variance = np.var(gray)
        if variance > 100:  # Lower threshold
            score += 1
        
        # 3. Color diversity (figures often have multiple colors)
        if len(img.shape) == 3:
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors > 20:  # Lower threshold
                score += 1
        
        # 4. Text content analysis (expanded regions may have more text)
        text_likelihood = self._estimate_text_content(gray)
        if text_likelihood < 0.8:  # Higher threshold (more text allowed)
            score += 1
        
        # 5. Geometric shapes detection
        if self._detect_geometric_patterns(gray):
            score += 1
        
        # Require at least 2 positive indicators (lower threshold)
        return score >= 2

    # Caption detection and validation methods
    async def _detect_captions(self, candidates: List[FigureCandidate], 
                             pdf_path: Path) -> List[FigureCandidate]:
        """Enhanced caption detection for figure candidates"""
        doc = fitz.open(str(pdf_path))
        
        for candidate in candidates:
            try:
                page = doc[candidate.page - 1]
                
                # Get text around figure
                nearby_text = self._get_text_near_figure(page, candidate.bbox)
                
                # Extract caption using patterns
                caption, label = self._extract_caption_from_text(nearby_text)
                
                if caption:
                    candidate.caption = caption
                if label and not candidate.label:
                    candidate.label = label
                    
            except Exception as e:
                logger.warning(f"Caption detection failed for candidate on page {candidate.page}: {e}")
        
        doc.close()
        return candidates
    
    def _get_text_near_figure(self, page, figure_bbox: BoundingBox, 
                            proximity: float = 100) -> List[Dict[str, Any]]:
        """Get text blocks near figure"""
        text_dict = page.get_text("dict")
        nearby_text = []
        
        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # Text block
                block_bbox = block["bbox"]
                
                # Check if block is near figure
                if self._is_text_near_figure(block_bbox, figure_bbox, proximity):
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    nearby_text.append({
                        'text': block_text.strip(),
                        'bbox': block_bbox,
                        'distance': self._calculate_bbox_distance(block_bbox, figure_bbox)
                    })
        
        # Sort by distance to figure
        nearby_text.sort(key=lambda x: x['distance'])
        return nearby_text
    
    def _is_text_near_figure(self, text_bbox: List[float], figure_bbox: BoundingBox, 
                           proximity: float) -> bool:
        """Check if text is near figure"""
        distance = self._calculate_bbox_distance(text_bbox, figure_bbox)
        return distance <= proximity
    
    def _calculate_bbox_distance(self, text_bbox: List[float], 
                               figure_bbox: BoundingBox) -> float:
        """Calculate distance between bounding boxes"""
        # Calculate center points
        text_center_x = (text_bbox[0] + text_bbox[2]) / 2
        text_center_y = (text_bbox[1] + text_bbox[3]) / 2
        
        fig_center_x = (figure_bbox.x1 + figure_bbox.x2) / 2
        fig_center_y = (figure_bbox.y1 + figure_bbox.y2) / 2
        
        # Euclidean distance
        return np.sqrt((text_center_x - fig_center_x)**2 + 
                      (text_center_y - fig_center_y)**2)
    
    def _extract_caption_from_text(self, nearby_text: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
        """Extract caption and label from nearby text"""
        caption = None
        label = None
        
        for text_block in nearby_text:
            text = text_block['text']
            
            for pattern in self.CAPTION_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    if len(match.groups()) >= 2:
                        figure_num = match.group(1)
                        caption_text = match.group(2).strip()
                        
                        # Clean caption text
                        caption_text = re.sub(r'\s+', ' ', caption_text)
                        caption_text = caption_text.strip('.,;:')
                        
                        if caption_text and len(caption_text) > 5:  # Reasonable caption length
                            label = f"Figure {figure_num}"
                            caption = caption_text
                            return caption, label
        
        return caption, label
    
    async def _validate_figure_candidates(self, candidates: List[FigureCandidate], 
                                        pdf_path: Path) -> List[FigureCandidate]:
        """Validate figure candidates using multiple criteria"""
        validated = []
        
        for candidate in candidates:
            validation_score = await self._calculate_figure_validation_score(candidate, pdf_path)
            
            if validation_score > 0.6:  # Validation threshold
                candidate.confidence = (candidate.confidence + validation_score) / 2
                validated.append(candidate)
                logger.debug(f"Figure validated: {candidate.label}, score: {validation_score:.2f}")
            else:
                logger.debug(f"Figure rejected: {candidate.label}, score: {validation_score:.2f}")
        
        return validated
    
    async def _calculate_figure_validation_score(self, candidate: FigureCandidate, 
                                               pdf_path: Path) -> float:
        """Calculate comprehensive validation score for figure"""
        score = 0.0
        
        # 1. Size and proportion validation (0.2 weight)
        bbox_score = self._validate_figure_bbox(candidate.bbox)
        score += 0.2 * bbox_score
        
        # 2. Caption quality (0.3 weight)
        caption_score = self._validate_caption_quality(candidate.caption, candidate.label)
        score += 0.3 * caption_score
        
        # 3. Image content validation (0.3 weight)
        if candidate.image_path and Path(candidate.image_path).exists():
            content_score = self._validate_image_content(candidate.image_path)
            score += 0.3 * content_score
        else:
            score += 0.15  # Partial score if no image available
        
        # 4. Method confidence (0.2 weight)
        method_score = self._get_method_confidence_score(candidate.method)
        score += 0.2 * method_score
        
        return min(1.0, score)
    
    async def _extract_ocr_from_figure(self, candidate: FigureCandidate, pdf_path: Path) -> FigureCandidate:
        """Extract OCR text from figure image for LLM processing"""
        if not self._is_ocr_available() or not candidate.image_path:
            return candidate
        
        try:
            # Load the figure image
            image_path = Path(candidate.image_path)
            if not image_path.exists():
                return candidate
            
            # Extract text using OCR manager
            ocr_result = await self.ocr_manager.extract_text(image_path)
            
            if ocr_result and ocr_result.text.strip():
                candidate.ocr_text = ocr_result.text
                candidate.ocr_confidence = ocr_result.confidence
                
                logger.debug(f"OCR extracted from figure {candidate.label}: {len(ocr_result.text)} chars, confidence: {ocr_result.confidence:.2f}, provider: {ocr_result.provider}")
            
        except Exception as e:
            logger.warning(f"OCR extraction failed for figure {candidate.label}: {e}")
        
        return candidate
    

    
    async def _extract_ocr_from_pdf_region(self, candidate: FigureCandidate, pdf_path: Path) -> FigureCandidate:
        """Extract OCR text from PDF region when image file is not available"""
        if not self._is_ocr_available():
            return candidate
        
        try:
            # Open PDF and extract the region as image
            doc = fitz.open(str(pdf_path))
            page = doc[candidate.page - 1]  # Convert to 0-based index
            
            # Create transformation matrix for the figure region
            bbox = candidate.bbox
            rect = fitz.Rect(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            
            # Extract region as image with higher resolution
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat, clip=rect)
            
            # Convert to bytes
            img_data = pix.tobytes("png")
            
            # Extract text using OCR manager
            ocr_result = await self.ocr_manager.extract_text_from_bytes(img_data)
            
            if ocr_result and ocr_result.text.strip():
                candidate.ocr_text = ocr_result.text
                candidate.ocr_confidence = ocr_result.confidence
                
                logger.debug(f"OCR extracted from PDF region {candidate.label}: {len(ocr_result.text)} chars, confidence: {ocr_result.confidence:.2f}, provider: {ocr_result.provider}")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PDF region OCR extraction failed for figure {candidate.label}: {e}")
        
        return candidate
    
    def _validate_figure_bbox(self, bbox: BoundingBox) -> float:
        """Validate figure bounding box dimensions"""
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        score = 0.0
        
        # Size validation
        if self.MIN_FIGURE_AREA <= area <= self.MAX_FIGURE_AREA:
            score += 0.5
        
        # Aspect ratio validation
        if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
            score += 0.5
        
        return score
    
    def _validate_caption_quality(self, caption: Optional[str], 
                                label: Optional[str]) -> float:
        """Validate quality of detected caption"""
        score = 0.0
        
        # Label validation
        if label and re.match(r'(?:Figure|Fig\.?)\s*\d+', label, re.IGNORECASE):
            score += 0.4
        
        # Caption validation
        if caption:
            # Length check
            if 10 <= len(caption) <= 500:  # Reasonable caption length
                score += 0.3
            
            # Content check (should be descriptive)
            descriptive_words = ['shows', 'displays', 'presents', 'illustrates', 
                               'depicts', 'comparison', 'results', 'data']
            if any(word in caption.lower() for word in descriptive_words):
                score += 0.2
            
            # Sentence structure
            if caption.endswith('.') and len(caption.split()) >= 3:
                score += 0.1
        
        return score
    
    def _validate_image_content(self, image_path: str) -> float:
        """Validate actual image content"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0.0
            
            # Reuse the figure content validation
            return 1.0 if self._is_likely_figure_content(img) else 0.3
            
        except Exception as e:
            logger.warning(f"Image validation failed for {image_path}: {e}")
            return 0.5  # Neutral score if validation fails
    
    def _get_method_confidence_score(self, method: str) -> float:
        """Get confidence score based on extraction method"""
        method_scores = {
            'pdfplumber': 0.9,
            'pymupdf': 0.85,
            'cv_contour': 0.7,
            'chart_axes': 0.88,
            'chart_axes_grouped': 0.9,
        }
        return method_scores.get(method, 0.5)
    
    async def _match_text_references(self, candidates: List[FigureCandidate], 
                                   pdf_path: Path) -> List[FigureCandidate]:
        """Match figures with text references"""
        # Find all figure references in document
        doc = fitz.open(str(pdf_path))
        all_references = {}
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            
            # Find figure references
            references = re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', page_text, re.IGNORECASE)
            for ref in references:
                if ref not in all_references:
                    all_references[ref] = []
                all_references[ref].append(page_num)
        
        doc.close()
        
        # Match candidates with references
        for candidate in candidates:
            if candidate.label:
                fig_num = re.search(r'\d+', candidate.label)
                if fig_num:
                    num = fig_num.group()
                    if num in all_references:
                        candidate.confidence += 0.1  # Boost confidence for referenced figures
                        # Store reference information
                        if not hasattr(candidate, 'references'):
                            candidate.references = []
                        # Convert references to strings
                        candidate.references.extend([str(ref) for ref in all_references[num]])
        
        return candidates
    
    def _deduplicate_and_rank_figures(self, candidates: List[FigureCandidate]) -> List[Figure]:
        """Deduplicate candidates and convert to final Figure objects"""
        # Group by page
        page_candidates = {}
        for candidate in candidates:
            if candidate.page not in page_candidates:
                page_candidates[candidate.page] = []
            page_candidates[candidate.page].append(candidate)
        
        final_figures = []
        
        for page_num, page_cands in page_candidates.items():
            # Remove duplicates within page
            unique_cands = self._remove_duplicate_candidates(page_cands)
            
            # Sort by confidence
            unique_cands.sort(key=lambda x: x.confidence, reverse=True)
            
            # Convert to Figure objects
            for idx, candidate in enumerate(unique_cands):
                figure = Figure(
                    label=candidate.label or f"Figure {page_num}.{idx}",
                    caption=candidate.caption,
                    page=candidate.page,
                    bbox=candidate.bbox,
                    image_path=candidate.image_path,
                    type=self._determine_figure_type(candidate),
                    references=getattr(candidate, 'references', []),
                    extraction_method=candidate.method,
                    confidence=candidate.confidence,
                    ocr_text=candidate.ocr_text,  # Include OCR extracted text
                    ocr_confidence=candidate.ocr_confidence  # Include OCR confidence
                )
                final_figures.append(figure)
        
        return final_figures
    
    def _remove_duplicate_candidates(self, candidates: List[FigureCandidate], 
                                   iou_threshold: float = 0.5) -> List[FigureCandidate]:
        """Remove duplicate candidates based on bbox overlap"""
        unique_candidates = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for unique_cand in unique_candidates:
                iou = self._calculate_bbox_iou(candidate.bbox, unique_cand.bbox)
                if iou > iou_threshold:
                    # Keep the one with higher confidence
                    if candidate.confidence > unique_cand.confidence:
                        unique_candidates.remove(unique_cand)
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _calculate_bbox_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate IoU between two bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_figure_type(self, candidate: FigureCandidate) -> str:
        """Determine figure type based on content and method"""
        if 'chart' in candidate.method:
            return 'chart'
        elif 'diagram' in candidate.method:
            return 'diagram'
        else:
            return 'figure'
    
    def _deduplicate_candidates(self, candidates: List[FigureCandidate]) -> List[FigureCandidate]:
        """Remove duplicate figure candidates based on bounding box overlap"""
        if not candidates:
            return []
        
        unique_candidates = []
        for candidate in candidates:
            is_duplicate = False
            for existing in unique_candidates:
                if self._calculate_overlap(candidate.bbox, existing.bbox) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def _validate_candidate(self, candidate: FigureCandidate) -> bool:
        """Validate figure candidate using page-relative thresholds (robust to DPI/page size)"""
        bbox = candidate.bbox
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        if w <= 0 or h <= 0:
            return False

        # Page-relative thresholds (robust to dpi / page size)
        page_w = candidate.page_width or 612  # fallback A4/letter-ish
        page_h = candidate.page_height or 792
        page_area = page_w * page_h
        area = w * h
        ar = w / h

        # Allow 0.5%–85% of page area for a single figure (multi-panels included)
        if not (0.005 * page_area <= area <= 0.85 * page_area):
            logger.debug(f"Candidate validation failed: area {area:.0f} not in range [{0.005 * page_area:.0f}, {0.85 * page_area:.0f}]")
            return False
        
        # Aspect ratio guardrails (very long strips rejected)
        if ar < 0.15 or ar > 6.0:
            logger.debug(f"Candidate validation failed: aspect ratio {ar:.2f} not in range [0.15, 6.0]")
            return False
        
        return True
    
    def _create_figure_from_candidate(self, candidate: FigureCandidate, index: int) -> Figure:
        """Create Figure object from validated candidate"""
        return Figure(
            id=f"figure_{candidate.page}_{index}",
            label=candidate.label or f"Figure {candidate.page}.{index}",
            caption=candidate.caption,
            page=candidate.page,
            bbox=candidate.bbox,
            extraction_method=candidate.method,
            confidence=candidate.confidence,
            image_path=candidate.image_path,
            # Include OCR extracted text from the figure image
            ocr_text=candidate.ocr_text,
            ocr_confidence=candidate.ocr_confidence
        )