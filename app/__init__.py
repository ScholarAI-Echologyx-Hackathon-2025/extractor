# app/__init__.py
"""PDF Extractor Application Package"""
from app.config import settings

__version__ = settings.app_version
__all__ = ['settings']