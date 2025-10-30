# services/extractors/code_extractor.py
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import pygments
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from app.models.schemas import CodeBlock, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError


class CodeExtractor:
    """
    Extract code blocks from PDFs using multiple techniques:
    1. Text pattern matching
    2. Visual detection (indentation, monospace fonts)
    3. Syntax highlighting detection
    4. Algorithm/pseudocode detection
    """
    
    # Common programming languages in academic papers
    LANGUAGES = [
        'python', 'java', 'c', 'cpp', 'javascript', 'r', 'matlab',
        'latex', 'bash', 'sql', 'julia', 'scala', 'go', 'rust'
    ]
    
    # Code indicators
    CODE_PATTERNS = {
        'function_def': [
            r'def\s+\w+\s*\(',  # Python
            r'function\s+\w+\s*\(',  # JavaScript
            r'public\s+\w+\s+\w+\s*\(',  # Java
            r'void\s+\w+\s*\(',  # C/C++
            r'func\s+\w+\s*\(',  # Go
            r'fn\s+\w+\s*\(',  # Rust
        ],
        'control_flow': [
            r'\bif\s*\(',
            r'\bfor\s*\(',
            r'\bwhile\s*\(',
            r'\bswitch\s*\(',
            r'\btry\s*\{',
            r'\bcatch\s*\(',
        ],
        'imports': [
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'#include\s*<',
            r'using\s+namespace',
            r'require\(',
        ],
        'comments': [
            r'//\s*\w+',  # Single line
            r'/\*.*?\*/',  # Multi-line
            r'#\s*\w+',  # Python/Bash
            r'%\s*\w+',  # Matlab/LaTeX
        ],
        'algorithms': [
            r'Algorithm\s+\d+',
            r'Input:',
            r'Output:',
            r'Require:',
            r'Ensure:',
            r'procedure\s+\w+',
            r'end\s+procedure',
        ]
    }
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "code"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    async def extract(self, pdf_path: Path) -> List[CodeBlock]:
        """Extract code blocks from PDF"""
        all_code_blocks = []
        
        # Method 1: Text-based extraction
        try:
            text_blocks = await self._extract_from_text(pdf_path)
            all_code_blocks.extend(text_blocks)
            logger.info(f"Text extraction found {len(text_blocks)} code blocks")
        except Exception as e:
            logger.error(f"Text-based code extraction failed: {e}")
        
        # Method 2: Visual detection
        try:
            visual_blocks = await self._extract_visual_code(pdf_path)
            new_blocks = self._deduplicate_code_blocks(all_code_blocks, visual_blocks)
            all_code_blocks.extend(new_blocks)
            logger.info(f"Visual detection found {len(new_blocks)} additional code blocks")
        except Exception as e:
            logger.error(f"Visual code extraction failed: {e}")
        
        # Method 3: Algorithm/Pseudocode extraction
        try:
            algo_blocks = await self._extract_algorithms(pdf_path)
            new_blocks = self._deduplicate_code_blocks(all_code_blocks, algo_blocks)
            all_code_blocks.extend(new_blocks)
            logger.info(f"Algorithm extraction found {len(new_blocks)} blocks")
        except Exception as e:
            logger.error(f"Algorithm extraction failed: {e}")
        
        # Post-process: detect language for blocks without it
        for block in all_code_blocks:
            if not block.language:
                block.language = self._detect_language(block.code)
        
        return all_code_blocks
    
    async def _extract_from_text(self, pdf_path: Path) -> List[CodeBlock]:
        """Extract code blocks from PDF text"""
        code_blocks = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Get page text with formatting
            blocks = page.get_text("dict")
            
            # Find code blocks
            page_code_blocks = self._find_code_in_blocks(blocks, page_num)
            code_blocks.extend(page_code_blocks)
            
            # Also check for inline code
            text = page.get_text()
            inline_code = self._find_inline_code(text, page_num)
            code_blocks.extend(inline_code)
        
        doc.close()
        return code_blocks
    
    def _find_code_in_blocks(self, blocks: Dict, page_num: int) -> List[CodeBlock]:
        """Find code blocks in page blocks"""
        code_blocks = []
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 0:  # Text block
                lines = []
                bbox = None
                is_code = False
                
                # Collect lines and check for code patterns
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        line_text += text
                        
                        # Check font (monospace fonts indicate code)
                        font = span.get("font", "")
                        if any(mono in font.lower() for mono in ['mono', 'courier', 'consolas']):
                            is_code = True
                        
                        # Update bbox
                        span_bbox = span.get("bbox")
                        if span_bbox and len(span_bbox) == 4:
                            if bbox is None:
                                bbox = list(span_bbox)
                            else:
                                bbox[0] = min(bbox[0], span_bbox[0])
                                bbox[1] = min(bbox[1], span_bbox[1])
                                bbox[2] = max(bbox[2], span_bbox[2])
                                bbox[3] = max(bbox[3], span_bbox[3])
                    
                    lines.append(line_text)
                
                # Check if block contains code
                block_text = "\n".join(lines)
                if is_code or self._is_code_block(block_text):
                    # Clean and format code
                    code = self._clean_code(block_text)
                    
                    if len(code) > 10:  # Minimum code length
                        code_block = CodeBlock(
                            code=code,
                            page=page_num,
                            language=self._detect_language(code),
                            bbox=BoundingBox(
                                x1=bbox[0] if bbox else 0,
                                y1=bbox[1] if bbox else 0,
                                x2=bbox[2] if bbox else 100,
                                y2=bbox[3] if bbox else 100,
                                page=page_num
                            ) if bbox else None
                        )
                        code_blocks.append(code_block)
        
        return code_blocks
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text is likely a code block with enhanced validation"""
        if len(text) < 20:  # Increased minimum length
            return False
        
        # Check for code indicators
        code_indicators = 0
        total_checks = 0
        
        # 1. Check for programming keywords
        programming_keywords = [
            'def', 'function', 'class', 'if', 'else', 'for', 'while', 'return',
            'import', 'from', 'include', 'using', 'public', 'private', 'void',
            'int', 'float', 'string', 'bool', 'true', 'false', 'null', 'var',
            'const', 'let', 'try', 'catch', 'finally', 'throw', 'new', 'this',
            'super', 'extends', 'implements', 'interface', 'enum', 'switch',
            'case', 'break', 'continue', 'do', 'while', 'for', 'in', 'of'
        ]
        
        text_lower = text.lower()
        for keyword in programming_keywords:
            if keyword in text_lower:
                code_indicators += 1
                break  # Only count once per keyword type
        
        total_checks += 1
        
        # 2. Check for code patterns
        for pattern_type, patterns in self.CODE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    code_indicators += 1
                    break  # Only count once per pattern type
            total_checks += 1
        
        # 3. Check for indentation patterns (common in code)
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > len(lines) * 0.3:  # More than 30% lines are indented
            code_indicators += 1
        total_checks += 1
        
        # 4. Check for balanced brackets/parentheses
        open_brackets = text.count('(') + text.count('[') + text.count('{')
        close_brackets = text.count(')') + text.count(']') + text.count('}')
        if abs(open_brackets - close_brackets) <= 2:  # Roughly balanced
            code_indicators += 1
        total_checks += 1
        
        # 5. Check for semicolons and assignment operators
        if text.count(';') > len(lines) * 0.5 or text.count('=') > len(lines) * 0.3:
            code_indicators += 1
        total_checks += 1
        
        # 6. Check for comments
        comment_patterns = [r'//', r'/\*', r'#', r'%', r'<!--']
        for pattern in comment_patterns:
            if re.search(pattern, text):
                code_indicators += 1
                break
        total_checks += 1
        
        # 7. Check for function calls (parentheses with text)
        function_calls = re.findall(r'\w+\s*\([^)]*\)', text)
        if len(function_calls) > len(lines) * 0.2:
            code_indicators += 1
        total_checks += 1
        
        # 8. Check for variable assignments
        assignments = re.findall(r'\w+\s*=\s*[^;]+', text)
        if len(assignments) > len(lines) * 0.2:
            code_indicators += 1
        total_checks += 1
        
        # Require at least 60% of checks to pass for code classification
        return (code_indicators / total_checks) >= 0.6
        
        # Count code indicators
        score = 0
        
        # Check for code patterns
        for pattern_type, patterns in self.CODE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE):
                    score += 2 if pattern_type in ['function_def', 'imports'] else 1
        
        # Check for indentation (common in code)
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith((' ', '\t')))
        if indented_lines > len(lines) * 0.3:
            score += 2
        
        # Check for common code characters
        code_chars = ['{', '}', '[', ']', '(', ')', ';', '=', '+', '-', '*', '/', '%']
        char_count = sum(text.count(char) for char in code_chars)
        if char_count > len(text) * 0.05:
            score += 1
        
        # Check for line numbers
        if re.search(r'^\s*\d+[:\.\s]', text, re.MULTILINE):
            score += 2
        
        return score >= 3
    
    def _find_inline_code(self, text: str, page_num: int) -> List[CodeBlock]:
        """Find inline code snippets"""
        code_blocks = []
        
        # Look for backtick-enclosed code
        backtick_pattern = r'`([^`]+)`'
        for match in re.finditer(backtick_pattern, text):
            code = match.group(1)
            if len(code) > 5:
                code_blocks.append(CodeBlock(
                    code=code,
                    page=page_num,
                    language=self._detect_language(code),
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                ))
        
        # Look for verbatim environments (LaTeX)
        verbatim_pattern = r'\\begin{verbatim}(.*?)\\end{verbatim}'
        for match in re.finditer(verbatim_pattern, text, re.DOTALL):
            code = match.group(1).strip()
            if code:
                code_blocks.append(CodeBlock(
                    code=code,
                    page=page_num,
                    language=self._detect_language(code)
                ))
        
        # Look for listing environments (LaTeX)
        listing_pattern = r'\\begin{lstlisting}(?:\[.*?\])?(.*?)\\end{lstlisting}'
        for match in re.finditer(listing_pattern, text, re.DOTALL):
            code = match.group(1).strip()
            if code:
                code_blocks.append(CodeBlock(
                    code=code,
                    page=page_num,
                    language=self._detect_language(code)
                ))
        
        return code_blocks
    
    async def _extract_visual_code(self, pdf_path: Path) -> List[CodeBlock]:
        """Detect code blocks visually"""
        code_blocks = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect code regions
            code_regions = self._detect_code_regions(gray)
            
            for region in code_regions:
                x, y, w, h = region
                
                # Extract text from region
                roi = img[y:y+h, x:x+w]
                text = self._extract_text_from_image(roi)
                
                if text and self._is_code_block(text):
                    # Convert coordinates to PDF space
                    scale = page.rect.width / img.shape[1]
                    
                    code_block = CodeBlock(
                        code=self._clean_code(text),
                        page=page_num,
                        language=self._detect_language(text),
                        bbox=BoundingBox(
                            x1=x * scale,
                            y1=y * scale,
                            x2=(x + w) * scale,
                            y2=(y + h) * scale,
                            page=page_num,
                            confidence=0.7
                        )
                    )
                    code_blocks.append(code_block)
        
        doc.close()
        return code_blocks
    
    def _detect_code_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that likely contain code"""
        regions = []
        
        # Apply edge detection
        edges = cv2.Canny(img, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = img.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if (w > width * 0.3 and  # At least 30% of page width
                h > 30 and  # Minimum height
                h < height * 0.5):  # Not more than half page
                
                # Check if region has code-like characteristics
                roi = img[y:y+h, x:x+w]
                if self._has_code_characteristics(roi):
                    regions.append((x, y, w, h))
        
        return regions
    
    def _has_code_characteristics(self, img: np.ndarray) -> bool:
        """Check if image region has code-like characteristics"""
        # Check for consistent line spacing (common in code)
        edges = cv2.Canny(img, 50, 150)
        
        # Horizontal projection
        horizontal_proj = np.sum(edges, axis=1)
        
        # Find peaks (text lines)
        peaks = []
        threshold = np.max(horizontal_proj) * 0.1
        for i, val in enumerate(horizontal_proj):
            if val > threshold:
                peaks.append(i)
        
        if len(peaks) < 3:
            return False
        
        # Check for consistent spacing
        spacings = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        if spacings:
            avg_spacing = np.mean(spacings)
            std_spacing = np.std(spacings)
            
            # Code typically has consistent line spacing
            if std_spacing < avg_spacing * 0.3:
                return True
        
        return False
    
    def _extract_text_from_image(self, img: np.ndarray) -> str:
        """Extract text from image using OCR"""
        try:
            import pytesseract
            
            # Preprocess image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with code-friendly settings
            text = pytesseract.image_to_string(
                binary,
                config='--psm 6 preserve_interword_spaces=1'
            )
            
            return text
        except:
            return ""
    
    async def _extract_algorithms(self, pdf_path: Path) -> List[CodeBlock]:
        """Extract algorithm/pseudocode blocks"""
        code_blocks = []
        doc = fitz.open(str(pdf_path))
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            
            # Find algorithm blocks
            algo_pattern = r'Algorithm\s+\d+[^\n]*\n((?:.*\n)*?)(?:^$|\n\n)'
            for match in re.finditer(algo_pattern, text, re.MULTILINE):
                algo_text = match.group(0)
                
                # Clean and format
                algo_code = self._format_algorithm(algo_text)
                
                if algo_code:
                    code_block = CodeBlock(
                        code=algo_code,
                        page=page_num,
                        language="pseudocode",
                        context=match.group(0)[:100]
                    )
                    code_blocks.append(code_block)
            
            # Find procedure blocks
            proc_pattern = r'procedure\s+(\w+).*?end\s+procedure'
            for match in re.finditer(proc_pattern, text, re.DOTALL | re.IGNORECASE):
                proc_code = match.group(0)
                
                code_block = CodeBlock(
                    code=proc_code,
                    page=page_num,
                    language="pseudocode"
                )
                code_blocks.append(code_block)
        
        doc.close()
        return code_blocks
    
    def _format_algorithm(self, text: str) -> str:
        """Format algorithm text"""
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            # Remove line numbers if present
            line = re.sub(r'^\s*\d+[:\.\s]', '', line)
            
            # Preserve indentation
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            if stripped:
                formatted.append(' ' * indent + stripped)
        
        return '\n'.join(formatted)
    
    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language of code"""
        try:
            # Try to guess using Pygments
            lexer = guess_lexer(code)
            return lexer.aliases[0] if lexer.aliases else None
        except:
            pass
        
        # Fallback to pattern matching
        language_patterns = {
            'python': [r'def\s+\w+\(', r'import\s+\w+', r'from\s+\w+\s+import'],
            'java': [r'public\s+class', r'public\s+static\s+void\s+main'],
            'javascript': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'let\s+\w+\s*='],
            'c': [r'#include\s*<', r'int\s+main\('],
            'cpp': [r'#include\s*<', r'using\s+namespace\s+std'],
            'r': [r'<-', r'library\(', r'data\.frame\('],
            'matlab': [r'function\s+\[', r'end\s*$', r'%\s+\w+'],
            'latex': [r'\\begin\{', r'\\end\{', r'\\section\{'],
        }
        
        for lang, patterns in language_patterns.items():
            if any(re.search(pattern, code) for pattern in patterns):
                return lang
        
        return None
    
    def _clean_code(self, code: str) -> str:
        """Clean and format code"""
        # Remove line numbers
        lines = code.split('\n')
        cleaned = []
        
        for line in lines:
            # Remove common line number patterns
            line = re.sub(r'^\s*\d+[:\.\s]\s*', '', line)
            
            # Remove trailing whitespace
            line = line.rstrip()
            
            cleaned.append(line)
        
        # Remove excessive blank lines
        result = '\n'.join(cleaned)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _deduplicate_code_blocks(self,
                                existing: List[CodeBlock],
                                new: List[CodeBlock]) -> List[CodeBlock]:
        """Remove duplicate code blocks"""
        unique_new = []
        
        for new_block in new:
            is_duplicate = False
            
            for exist_block in existing:
                if new_block.page == exist_block.page:
                    # Check code similarity
                    similarity = self._code_similarity(new_block.code, exist_block.code)
                    if similarity > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_new.append(new_block)
        
        return unique_new
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code blocks"""
        # Simple token-based similarity
        tokens1 = set(re.findall(r'\w+', code1.lower()))
        tokens2 = set(re.findall(r'\w+', code2.lower()))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0