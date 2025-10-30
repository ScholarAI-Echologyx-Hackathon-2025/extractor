# services/ocr/ocr_manager.py
import logging
import asyncio
import httpx
import base64
from typing import Optional, List, Dict, Any
from pathlib import Path
from app.config import settings
from app.utils.helpers import async_retry

logger = logging.getLogger(__name__)


class OCRManager:
    """
    OCR Manager using OCR.space API
    Provides OCR functionality for image and PDF processing
    """
    
    def __init__(self, **kwargs):
        self.api_key = getattr(settings, 'ocr_space_api_key', 'helloworld')  # Default free key
        self.api_url = "https://api.ocr.space/parse/image"
        self.providers = ["ocr_space"]
        self.primary_provider = "ocr_space"
        logger.info("OCR Manager initialized with OCR.space API")
    
    def _initialize_providers(self, **kwargs):
        """Initialize OCR providers"""
        logger.info("OCR.space provider initialized")
    
    async def extract_text(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Extract text from image file using OCR.space API"""
        try:
            if not settings.use_ocr:
                logger.info("OCR disabled in config - skipping text extraction")
                return None
            
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Read image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            return await self.extract_text_from_bytes(image_bytes)
            
        except Exception as e:
            logger.error(f"OCR text extraction failed for {image_path}: {e}")
        return None
    
    def _preprocess_image_bytes(self, image_bytes: bytes) -> bytes:
        """Optionally preprocess image bytes to improve OCR quality.
        - Convert to grayscale
        - Slight contrast enhancement
        - Upscale small images to ~1200px on the longer side
        Returns PNG bytes.
        """
        if not getattr(settings, 'ocr_preprocess', True):
            return image_bytes
        try:
            from PIL import Image, ImageOps, ImageFilter
            import io
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert('L')  # grayscale
                # auto-contrast and slight sharpening
                img = ImageOps.autocontrast(img)
                img = img.filter(ImageFilter.SHARPEN)
                # upscale if needed
                w, h = img.size
                longer = max(w, h)
                target = 1200
                if longer < target:
                    scale = target / float(longer)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = img.resize((new_w, new_h))
                out = io.BytesIO()
                img.save(out, format='PNG')
                return out.getvalue()
        except Exception as e:
            logger.debug(f"OCR preprocess skipped due to error: {e}")
            return image_bytes
    
    async def extract_text_from_bytes(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Extract text from image bytes using OCR.space API"""
        try:
            if not settings.use_ocr:
                logger.info("OCR disabled in config - skipping text extraction")
                return None
            
            # Optional preprocessing
            processed_bytes = self._preprocess_image_bytes(image_bytes)
            
            # Encode image to base64 with proper format
            base64_image = base64.b64encode(processed_bytes).decode('utf-8')
            
            # Determine file type from image bytes
            import imghdr
            file_type = imghdr.what(None, h=processed_bytes[:32])
            if file_type == 'jpeg':
                file_type = 'jpg'
            elif file_type is None:
                file_type = 'png'  # Default to PNG
            
            # Prepare API request data
            data = {
                'apikey': self.api_key,
                'base64Image': f'data:image/{file_type};base64,{base64_image}',
                'language': getattr(settings, 'ocr_language', 'eng'),
                'isOverlayRequired': 'true',  # Get word coordinates
                'detectOrientation': 'true',  # Auto-rotate if needed
                'scale': 'true',  # Upscale for better results
                'OCREngine': '2',  # Use Engine 2 for better accuracy
                'isTable': 'false'  # Not specifically for tables
            }
            
            return await self._post_with_retries(data)
            
        except Exception as e:
            logger.error(f"OCR.space API call failed: {e}")
            return None
    
    async def _post_with_retries(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Post to OCR.space with retry/backoff on transient errors."""
        timeout = getattr(settings, 'ocr_timeout', 30)
        max_retries = max(0, getattr(settings, 'ocr_max_retries', 2))

        async def do_post() -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.api_url, data=data)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("transient", request=response.request, response=response)
                response.raise_for_status()
                result = response.json()
                if result.get('IsErroredOnProcessing'):
                    err_msg = result.get('ErrorMessage')
                    logger.error(f"OCR.space API error: {err_msg}")
                    return {}
                return result

        try:
            result = await async_retry(
                do_post,
                max_retries=max_retries,
                backoff_initial=1.0,
                exceptions=(httpx.TimeoutException, httpx.HTTPError),
                on_error=lambda e, a: logger.debug(f"OCR post retry {a+1} due to {e}")
            )
            if not result:
                return None
            parsed_results = result.get('ParsedResults', [])
            if not parsed_results:
                logger.warning("No parsed results from OCR.space API")
                return None
            extracted_text = ""
            word_data: List[Dict[str, Any]] = []
            for parsed_result in parsed_results:
                if parsed_result.get('FileParseExitCode') == 1:
                    parsed_text = parsed_result.get('ParsedText', '')
                    extracted_text += parsed_text + "\n"
                    text_overlay = parsed_result.get('TextOverlay')
                    if text_overlay and text_overlay.get('HasOverlay'):
                        lines = text_overlay.get('Lines', [])
                        for line in lines:
                            words = line.get('Words', [])
                            for word in words:
                                word_data.append({
                                    'text': word.get('WordText', ''),
                                    'left': word.get('Left', 0),
                                    'top': word.get('Top', 0),
                                    'width': word.get('Width', 0),
                                    'height': word.get('Height', 0)
                                })
            return {
                'text': extracted_text.strip(),
                'words': word_data,
                'provider': 'ocr_space',
                'confidence': 0.8,
                'processing_time': result.get('ProcessingTimeInMilliseconds', 0)
            }
        except Exception as e:
            logger.error(f"OCR request failed: {e}")
            return None
    
    async def extract_text_from_url(self, image_url: str) -> Optional[Dict[str, Any]]:
        """Extract text from image URL directly using OCR.space API"""
        try:
            if not settings.use_ocr:
                logger.info("OCR disabled in config - skipping text extraction")
                return None
            
            data = {
                'apikey': self.api_key,
                'url': image_url,  # Direct URL instead of base64
                'language': getattr(settings, 'ocr_language', 'eng'),
                'isOverlayRequired': 'true',  # Get word coordinates
                'detectOrientation': 'true',  # Auto-rotate if needed
                'scale': 'true',  # Upscale for better results
                'OCREngine': '2',  # Use Engine 2 for better accuracy
                'isTable': 'false'  # Not specifically for tables
            }
            return await self._post_with_retries(data)
            
        except Exception as e:
            logger.error(f"OCR.space API call failed for URL {image_url}: {e}")
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get available OCR providers"""
        return self.providers if settings.use_ocr else []
    
    def is_any_provider_available(self) -> bool:
        """Check if any OCR provider is available"""
        return settings.use_ocr and len(self.providers) > 0
