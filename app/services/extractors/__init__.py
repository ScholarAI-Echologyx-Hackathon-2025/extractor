# services/extractors/__init__.py
"""Extractors Package"""
from app.services.extractors.grobid_extractor import GROBIDExtractor
from app.services.extractors.figure_extractor import FigureExtractor
# from app.services.extractors.table_extractor import TableExtractor  # Disabled for efficiency
from app.services.extractors.ocr_math_extractor import OCRMathExtractor
from app.services.extractors.code_extractor import CodeExtractor

__all__ = [
    'GROBIDExtractor',
    'FigureExtractor',
    # 'TableExtractor',   # Disabled for efficiency
    'OCRMathExtractor',
    'CodeExtractor'
]