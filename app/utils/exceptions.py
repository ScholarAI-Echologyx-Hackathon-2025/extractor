# utils/exceptions.py
from typing import Optional, Dict, Any


class PDFExtractorException(Exception):
    """Base exception for PDF extractor"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ExtractionError(PDFExtractorException):
    """Error during extraction process"""
    pass


class ServiceUnavailableError(PDFExtractorException):
    """External service is unavailable"""
    pass


class InvalidPDFError(PDFExtractorException):
    """Invalid or corrupted PDF file"""
    pass


class TimeoutError(PDFExtractorException):
    """Extraction timeout"""
    pass


class ConfigurationError(PDFExtractorException):
    """Configuration error"""
    pass