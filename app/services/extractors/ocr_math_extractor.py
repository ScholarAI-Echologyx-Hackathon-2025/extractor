"""
OCR and Math Formula Extractor (Memory Optimized)
Extracts mathematical formulas and equations from text content only.
No OCR capabilities - relies on pre-extracted text.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.models.schemas import Equation, BoundingBox
from app.services.extractors.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class OCRMathExtractor(BaseExtractor):
    """
    Memory-optimized OCR/Math extractor that processes pre-extracted text.
    No OCR capabilities - only text-based math detection.
    """

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.nougat_available = False
        self.tesseract_available = False
        logger.info("OCR/Math extractor initialized (OCR disabled for memory optimization)")
    
    async def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract mathematical formulas from pre-extracted text content.
        No OCR processing - only text-based math detection.
        """
        try:
            logger.info(f"Processing math extraction for: {pdf_path.name}")
            
            # Since we don't have OCR, we can only process if text is already extracted
            # This would typically be called after GROBID or other text extractors
            equations = []
            
            # Try to find any existing text files to process
            text_files = list(self.output_dir.glob("**/*.txt"))
            if text_files:
                logger.info(f"Found {len(text_files)} text files to process for math content")
                for text_file in text_files:
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        file_equations = self._extract_math_formulas(text_content, 0)
                        equations.extend(file_equations)
                    except Exception as e:
                        logger.warning(f"Failed to process text file {text_file}: {e}")
            else:
                logger.info("No pre-extracted text files found for math processing")
            
            logger.info(f"Extracted {len(equations)} mathematical formulas")
            
            return {
                "equations": equations,
                "total_count": len(equations),
                "extraction_method": "text_only"
            }
            
        except Exception as e:
            logger.error(f"Math extraction failed: {e}")
            return {
                "equations": [],
                "total_count": 0,
                "extraction_method": "text_only",
                "error": str(e)
            }

    def _extract_math_formulas(self, text: str, page_num: int) -> List[Equation]:
        """
        Extract mathematical formulas from text using pattern matching.
        No image processing - only text-based detection.
        """
        equations = []
        
        if not text or not text.strip():
            return equations
        
        # Basic math patterns for text-based detection
        math_patterns = [
            r'\$[^$]+\$',  # LaTeX inline math
            r'\\\[[^\]]*\\\]',  # LaTeX display math
            r'\\begin\{equation\}.*?\\end\{equation\}',  # LaTeX equation environment
            r'\\begin\{align\}.*?\\end\{align\}',  # LaTeX align environment
            r'[a-zA-Z]\s*=\s*[^=]+',  # Simple equations like "x = y + z"
            r'[a-zA-Z]\s*\+\s*[a-zA-Z]',  # Addition patterns
            r'[a-zA-Z]\s*\*\s*[a-zA-Z]',  # Multiplication patterns
            r'[a-zA-Z]\s*/\s*[a-zA-Z]',  # Division patterns
            r'[a-zA-Z]\s*\^\s*[0-9]',  # Exponentiation patterns
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                equation_text = match.group(0)
                
                # Create equation object
                equation = Equation(
                    id=f"eq_page{page_num + 1}_{len(equations)}",
                    page=page_num + 1,
                    text=equation_text,
                    type="latex" if "$" in equation_text or "\\" in equation_text else "text",
                    confidence=0.8,
                    bounding_box=BoundingBox(
                        x=0, y=0, width=100, height=50  # Default values
                    )
                )
                equations.append(equation)
        
        return equations
    
    def _validate_equation(self, equation: Equation) -> bool:
        """Validate extracted equation"""
        if not equation.text or len(equation.text.strip()) < 2:
            return False
        return True