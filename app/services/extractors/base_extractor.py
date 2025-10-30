"""
Base extractor class for all extraction services
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseExtractor(ABC):
    """
    Base class for all extractors
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    @abstractmethod
    async def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract content from PDF
        """
        pass
