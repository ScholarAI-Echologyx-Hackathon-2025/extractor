# utils/helpers.py
import hashlib
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
from loguru import logger


def validate_pdf(file_path: Path) -> bool:
    """
    Validate if file is a valid PDF
    """
    try:
        # Check file extension
        if file_path.suffix.lower() != '.pdf':
            return False
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type != 'application/pdf':
            logger.warning(f"Unexpected MIME type: {mime_type}")
        
        # Try to open with PyMuPDF
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        doc.close()
        
        if page_count == 0:
            return False
        
        return True
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        return False


def get_pdf_info(file_path: Path) -> Dict[str, Any]:
    """
    Get basic information about PDF
    """
    info = {
        'valid': False,
        'page_count': 0,
        'file_size': 0,
        'encrypted': False,
        'has_text': False,
        'has_images': False,
        'metadata': {}
    }
    
    try:
        # File size
        info['file_size'] = file_path.stat().st_size
        
        # Open PDF
        doc = fitz.open(str(file_path))
        
        info['valid'] = True
        info['page_count'] = len(doc)
        info['encrypted'] = doc.is_encrypted
        
        # Check for text and images
        for page in doc:
            if page.get_text().strip():
                info['has_text'] = True
            if page.get_images():
                info['has_images'] = True
            
            if info['has_text'] and info['has_images']:
                break
        
        # Extract metadata
        metadata = doc.metadata
        if metadata:
            info['metadata'] = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'created': str(metadata.get('creationDate', '')),
                'modified': str(metadata.get('modDate', ''))
            }
        
        doc.close()
        
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
    
    return info


def merge_bounding_boxes(boxes: List[Tuple[float, float, float, float]], 
                        threshold: float = 10) -> List[Tuple[float, float, float, float]]:
    """
    Merge nearby bounding boxes
    """
    if not boxes:
        return []
    
    # Sort by y-coordinate, then x-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    merged = []
    current = list(sorted_boxes[0])
    
    for box in sorted_boxes[1:]:
        # Check if boxes are close enough to merge
        if (abs(box[1] - current[3]) < threshold or  # Vertically adjacent
            (box[1] < current[3] and box[0] < current[2])):  # Overlapping
            # Merge boxes
            current[0] = min(current[0], box[0])  # x1
            current[1] = min(current[1], box[1])  # y1
            current[2] = max(current[2], box[2])  # x2
            current[3] = max(current[3], box[3])  # y2
        else:
            merged.append(tuple(current))
            current = list(box)
    
    merged.append(tuple(current))
    return merged


def calculate_reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate reading order for text blocks (handle multi-column layouts)
    """
    if not blocks:
        return []
    
    # Group blocks by approximate column
    columns = []
    for block in blocks:
        bbox = block.get('bbox', [0, 0, 0, 0])
        x_center = (bbox[0] + bbox[2]) / 2
        
        # Find which column this belongs to
        assigned = False
        for col in columns:
            col_center = col['center']
            if abs(x_center - col_center) < 50:  # Within 50 units
                col['blocks'].append(block)
                assigned = True
                break
        
        if not assigned:
            columns.append({
                'center': x_center,
                'blocks': [block]
            })
    
    # Sort columns left to right
    columns.sort(key=lambda c: c['center'])
    
    # Sort blocks within each column top to bottom
    ordered_blocks = []
    for col in columns:
        col['blocks'].sort(key=lambda b: b.get('bbox', [0, 0])[1])
        ordered_blocks.extend(col['blocks'])
    
    return ordered_blocks


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    replacements = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        '„': '"',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '…': '...',
        '–': '-',
        '—': '--',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'([.,;!?])(\w)', r'\1 \2', text)
    
    return text.strip()


def extract_page_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> List[Path]:
    """
    Extract all pages as images
    """
    from pdf2image import convert_from_path
    
    images = convert_from_path(str(pdf_path), dpi=dpi)
    
    image_paths = []
    for i, image in enumerate(images):
        image_path = output_dir / f"page_{i+1}.png"
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    
    return image_paths


def estimate_quality_score(extraction_result: Dict[str, Any]) -> float:
    """
    Estimate quality score of extraction
    """
    score = 0.0
    weights = {
        'metadata': 0.1,
        'sections': 0.3,
        'figures': 0.15,
        'tables': 0.15,
        'equations': 0.1,
        'code': 0.1,
        'references': 0.1
    }
    
    # Check metadata
    metadata = extraction_result.get('metadata', {})
    if metadata.get('title') and metadata.get('title') != 'Unknown':
        score += weights['metadata']
    
    # Check sections
    sections = extraction_result.get('sections', [])
    if sections:
        section_quality = min(1.0, len(sections) / 10)  # Expect ~10 sections
        score += weights['sections'] * section_quality
    
    # Check figures
    figures = extraction_result.get('figures', [])
    if figures:
        figure_quality = min(1.0, len(figures) / 5)  # Expect ~5 figures
        score += weights['figures'] * figure_quality
    
    # Check tables
    tables = extraction_result.get('tables', [])
    if tables:
        table_quality = min(1.0, len(tables) / 3)  # Expect ~3 tables
        score += weights['tables'] * table_quality
    
    # Check equations
    equations = extraction_result.get('equations', [])
    if equations:
        eq_quality = min(1.0, len(equations) / 10)  # Expect ~10 equations
        score += weights['equations'] * eq_quality
    
    # Check code
    code_blocks = extraction_result.get('code_blocks', [])
    if code_blocks:
        code_quality = min(1.0, len(code_blocks) / 3)  # Expect ~3 code blocks
        score += weights['code'] * code_quality
    
    # Check references
    references = extraction_result.get('references', [])
    if references:
        ref_quality = min(1.0, len(references) / 20)  # Expect ~20 references
        score += weights['references'] * ref_quality
    
    return round(score * 100, 2)


def create_extraction_summary(result: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of extraction results
    """
    summary = []
    
    summary.append("=== PDF Extraction Summary ===\n")
    
    # Metadata
    metadata = result.get('metadata', {})
    summary.append(f"Title: {metadata.get('title', 'Unknown')}")
    
    authors = metadata.get('authors', [])
    if authors:
        author_names = [a.get('name', '') for a in authors if isinstance(a, dict)]
        summary.append(f"Authors: {', '.join(author_names)}")
    
    summary.append(f"Year: {metadata.get('year', 'Unknown')}")
    
    # Statistics
    summary.append("\n=== Content Statistics ===")
    summary.append(f"Sections: {len(result.get('sections', []))}")
    summary.append(f"Figures: {len(result.get('figures', []))}")
    summary.append(f"Tables: {len(result.get('tables', []))}")
    summary.append(f"Equations: {len(result.get('equations', []))}")
    summary.append(f"Code Blocks: {len(result.get('code_blocks', []))}")
    summary.append(f"References: {len(result.get('references', []))}")
    summary.append(f"Entities: {len(result.get('entities', []))}")
    
    # Quality metrics
    summary.append("\n=== Quality Metrics ===")
    summary.append(f"Coverage: {result.get('extraction_coverage', 0)}%")
    
    confidence = result.get('confidence_scores', {})
    for key, value in confidence.items():
        summary.append(f"{key.capitalize()} Confidence: {value:.2f}")
    
    # Processing info
    summary.append("\n=== Processing Info ===")
    summary.append(f"Status: {result.get('status', 'Unknown')}")
    summary.append(f"Methods Used: {', '.join(result.get('extraction_methods', []))}")
    summary.append(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
    
    # Errors and warnings
    errors = result.get('errors', [])
    if errors:
        summary.append("\n=== Errors ===")
        for error in errors:
            summary.append(f"- {error}")
    
    warnings = result.get('warnings', [])
    if warnings:
        summary.append("\n=== Warnings ===")
        for warning in warnings:
            summary.append(f"- {warning}")
    
    return '\n'.join(summary)