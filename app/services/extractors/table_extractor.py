# services/extractors/table_extractor.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from loguru import logger
import json
import io
import re

from app.models.schemas import Table, BoundingBox
from app.config import settings
from app.utils.exceptions import ExtractionError
from app.services.cloudinary_service import cloudinary_service

from app.services.extractors.tabula_extractor import tabula_extractor


class TableExtractor:
    """
    Lightweight table extraction using rule-based methods only:
    1. PDFPlumber (primary method)
    2. Tabula (fallback only if PDFPlumber extracts 0 tables)
    
    AI models disabled for memory optimization
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.paper_folder / "tables"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Enhanced validation thresholds
        self.TABLE_CONFIDENCE_THRESHOLD = 0.5
        self.MIN_TABLE_ROWS = 2  # Minimum 2 rows (header + at least 1 data row)
        self.MIN_TABLE_COLS = 2  # Minimum 2 columns for a valid table
        self.MIN_CONTENT_DENSITY = 0.4  # At least 40% cells should have content
        self.MAX_CELL_TEXT_LENGTH = 300  # Maximum characters in a single cell (reduced from 500)
        self.MIN_NUMERIC_CELLS_RATIO = 0.05  # At least 5% cells should contain numbers for data tables (reduced from 0.1)
        
        # AI models disabled for memory optimization
        self.dl_models_available = False
        logger.info("Table extractor initialized (AI models disabled for memory optimization)")
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using lightweight rule-based methods only"""
        logger.info(f"Extracting tables from {pdf_path} using rule-based methods")
        
        all_tables = []
        
        # Method 1: PDFPlumber (primary method)
        try:
            pdfplumber_tables = await self._extract_with_pdfplumber(pdf_path)
            all_tables.extend(pdfplumber_tables)
            logger.info(f"PDFPlumber found {len(pdfplumber_tables)} valid tables")
        except Exception as e:
            logger.warning(f"PDFPlumber extraction failed: {e}")
        
        # Method 2: Tabula (fallback only if PDFPlumber extracts 0 tables)
        if len(pdfplumber_tables) == 0:
            try:
                tabula_tables = await tabula_extractor.extract(pdf_path)
                all_tables.extend(tabula_tables)
                logger.info(f"Tabula fallback found {len(tabula_tables)} tables")
            except Exception as e:
                logger.warning(f"Tabula fallback extraction failed: {e}")
        else:
            logger.info("PDFPlumber extracted tables, skipping Tabula fallback")
        
        # Deduplicate and validate tables with enhanced validation
        unique_tables = self._deduplicate_tables(all_tables)
        validated_tables = [table for table in unique_tables if self._validate_table_enhanced(table)]
        
        logger.info(f"Total valid tables after enhanced validation: {len(validated_tables)}")
        
        # Process and store tables
        final_tables = []
        for table in validated_tables:
            try:
                processed_table = await self._process_and_store_table(table, pdf_path)
                if processed_table:
                    final_tables.append(processed_table)
            except Exception as e:
                logger.error(f"Failed to process table {table.id}: {e}")
        
        logger.info(f"Successfully processed and stored {len(final_tables)} tables")
        return final_tables
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Table]:
        """Extract tables using PDFPlumber with optimized strategies"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Try line-based strategy first (most reliable for real tables)
                    strategies = [
                        {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},  # Line-based (best for bordered tables)
                        {'vertical_strategy': 'lines', 'horizontal_strategy': 'text'},   # Mixed strategy
                        {},  # Default strategy
                    ]
                    
                    page_tables_found = set()  # Track found tables to avoid duplicates
                    
                    for strategy_idx, table_settings in enumerate(strategies):
                        page_tables = page.extract_tables(table_settings=table_settings)
                    if page_tables:
                        for table_idx, table_data in enumerate(page_tables):
                                # Try to split if multiple tables are merged
                                split_tables = self._split_merged_tables(table_data)
                                
                                for split_idx, split_data in enumerate(split_tables):
                                    if self._is_valid_table_data_enhanced(split_data):
                                        # Create a hash of the table content to avoid duplicates
                                        table_hash = hash(str(split_data))
                                        if table_hash not in page_tables_found:
                                            page_tables_found.add(table_hash)
                                            # Create unique table index for split tables
                                            unique_idx = f"{table_idx}_{split_idx}" if len(split_tables) > 1 else str(table_idx)
                                table = self._create_table_from_data(
                                                split_data, page_num, 
                                                unique_idx, 
                                                f"pdfplumber_s{strategy_idx}", page
                                )
                                tables.append(table)
                                            logger.debug(f"PDFPlumber extracted valid table using strategy {strategy_idx}: {table.label}")
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
        
        return tables
    
    def _split_merged_tables(self, table_data):
        """Split merged tables based on empty rows or significant structure changes"""
        if not table_data or len(table_data) < 4:  # Too small to split
            return [table_data]
        
        # Look for natural break points
        segments = []
        current_segment = []
        
        for i in range(len(table_data)):
            row = table_data[i]
            
            # Check for empty separator rows or caption rows
            is_separator = False
            
            if not row or all(not str(cell).strip() for cell in row):
                is_separator = True
            else:
                row_text = " ".join(str(cell).strip() for cell in row if cell)
                if self._is_caption_or_reference(row_text):
                    is_separator = True
            
            if is_separator:
                # Save current segment if it has content
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                current_segment.append(row)
        
        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)
        
        # Filter segments to ensure each is a valid table candidate
        valid_tables = []
        for segment in segments:
            if len(segment) >= self.MIN_TABLE_ROWS:
                # Check column consistency within segment
                col_counts = [len(row) if row else 0 for row in segment]
                if col_counts:
                    mode_col_count = max(set(col_counts), key=col_counts.count)
                    consistent_rows = sum(1 for count in col_counts if abs(count - mode_col_count) <= 1)
                    consistency_ratio = consistent_rows / len(col_counts)
                    
                    # Only keep segments with good internal consistency
                    if consistency_ratio > 0.7:
                        valid_tables.append(segment)
        
        # If no valid splits found or only one segment, return original
        if len(valid_tables) <= 1:
            return [table_data]
        
        return valid_tables
    
    def _is_valid_table_data_enhanced(self, table_data) -> bool:
        """Enhanced validation to check if table data represents a real table"""
        if not table_data or not isinstance(table_data, (list, tuple)):
            return False
        
        # Pre-process: Check for and remove caption/figure text rows
        cleaned_data = self._remove_non_table_rows(table_data)
        
        # Use cleaned data for validation
        if not cleaned_data or len(cleaned_data) < self.MIN_TABLE_ROWS:
            return False
        
        if not cleaned_data[0] or len(cleaned_data[0]) < self.MIN_TABLE_COLS:
            return False
        
        # Enhanced validation checks
        
        # 1. Column consistency check - stronger requirement
        col_counts = [len(row) if row else 0 for row in cleaned_data]
        if col_counts:
            mode_col_count = max(set(col_counts), key=col_counts.count)
            consistent_rows = sum(1 for count in col_counts if count == mode_col_count)
            consistency_ratio = consistent_rows / len(col_counts)
            
            # Require at least 80% of rows to have consistent column count for valid tables
            if consistency_ratio < 0.8:
                logger.debug(f"Table rejected: Poor column consistency ({consistency_ratio:.2f})")
                return False
        
        # 2. Content density and pattern analysis
        content_cells = 0
        total_cells = 0
        numeric_cells = 0
        very_long_cells = 0
        empty_rows = 0
        prose_like_cells = 0
        
        for row_idx, row in enumerate(cleaned_data):
            if not row or all(not str(cell).strip() for cell in row):
                empty_rows += 1
                continue
                
            for cell in row:
                total_cells += 1
                cell_str = str(cell).strip() if cell else ""
                
                if cell_str:
                    content_cells += 1
                    
                    # Check for numeric content
                    if self._contains_number(cell_str):
                        numeric_cells += 1
                    
                    # Check for very long text (likely paragraph, not table cell)
                    if len(cell_str) > self.MAX_CELL_TEXT_LENGTH:
                        very_long_cells += 1
                    
                    # Check for prose-like text (sentences with punctuation)
                    if self._is_prose_text(cell_str):
                        prose_like_cells += 1
        
        # Reject if too many empty rows
        if empty_rows > len(cleaned_data) * 0.3:
            logger.debug("Table rejected: Too many empty rows")
            return False
        
        # Check content density
        if total_cells > 0:
            content_density = content_cells / total_cells
            if content_density < self.MIN_CONTENT_DENSITY:
                logger.debug(f"Table rejected: Low content density ({content_density:.2f})")
                return False
        
        # 3. Reject if contains too much prose-like text
        if prose_like_cells > total_cells * 0.3:  # More than 30% prose-like cells
            logger.debug("Table rejected: Contains too much prose text")
            return False
        
        # 4. Check for table-like patterns
        # Reject if cells contain very long text (paragraphs)
        if very_long_cells > total_cells * 0.15:  # More than 15% very long cells
            logger.debug("Table rejected: Contains paragraph-like text")
            return False
        
        # 5. Header validation - check if first row looks like headers
        if len(cleaned_data) >= 2:
            first_row = cleaned_data[0]
            second_row = cleaned_data[1] if len(cleaned_data) > 1 else []
            
            # Check if first row has reasonable header-like content
            header_like = self._is_header_row(first_row, second_row)
            
            # If it doesn't look like a header row and doesn't have numbers, likely not a table
            if not header_like and numeric_cells < 2:
                logger.debug("Table rejected: No clear headers and no numeric data")
                return False
        
        # 6. Check for minimum data table characteristics
        # A valid table should have either:
        # - Significant numeric data, OR
        # - Very consistent structure with clear headers
        
        has_numeric_data = (numeric_cells / max(content_cells, 1)) > self.MIN_NUMERIC_CELLS_RATIO
        has_consistent_structure = consistency_ratio > 0.9  # Very high consistency
        has_headers = self._is_header_row(cleaned_data[0], cleaned_data[1] if len(cleaned_data) > 1 else [])
        
        # More flexible validation: Accept tables that have good structure even with less numeric data
        if has_consistent_structure and has_headers:
            # Good structure with headers - likely a valid table
            return True
        elif has_numeric_data:
            # Has enough numeric data - likely a data table
            return True
        elif numeric_cells >= 2 and consistency_ratio > 0.8:
            # Has some numeric cells and reasonable consistency
            return True
        else:
            logger.debug("Table rejected: No numeric data and poor structure")
            return False
        
        # 7. Final check: Avoid single-column "tables"
        if len(cleaned_data[0]) <= 1:
            # Single column tables are often just formatted lists
            return False
        
        # 8. Check for minimum meaningful content
        if content_cells < 4:  # At least 4 cells with content
            logger.debug("Table rejected: Too little content")
            return False
        
        return True
    
    def _remove_non_table_rows(self, table_data):
        """Remove rows that are clearly not part of the table (captions, prose text)"""
        cleaned_data = []
        
        for row in table_data:
            if not row:
                continue
            
            # Remove completely empty columns from the row first
            cleaned_row = [cell for cell in row if cell is not None and str(cell).strip()]
            
            # Skip if row becomes empty after cleaning
            if not cleaned_row:
                continue
            
            # Check if row is likely a caption or prose text
            row_text = " ".join(str(cell).strip() for cell in cleaned_row)
            
            # Skip rows that are clearly figure captions or references
            if self._is_caption_or_reference(row_text):
                continue
            
            # Skip rows with too few non-empty cells relative to expected columns
            # (but keep the original row structure for consistency)
            non_empty_cells = sum(1 for cell in row if str(cell).strip())
            if len(row) > 3 and non_empty_cells == 1:
                # Single cell in a multi-column context - likely misaligned text
                if not self._contains_number(row_text):  # Unless it's a number
                    continue
            
            cleaned_data.append(row)
        
        return cleaned_data
    
    def _is_caption_or_reference(self, text: str) -> bool:
        """Check if text is likely a figure/table caption or reference"""
        text_lower = text.lower().strip()
        
        # Common caption patterns
        caption_patterns = [
            r'^figure\s+\d+[:\.]',
            r'^fig\.\s+\d+',
            r'^table\s+\d+[:\.]',
            r'^supplementary\s+(figure|table)',
            r'^\([a-z]\)',  # Sub-figure labels like (a), (b)
            r'^equation\s+\d+',
            r'^scheme\s+\d+',
        ]
        
        for pattern in caption_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check for reference-like text
        if text_lower.startswith(('see ', 'refer to ', 'as shown in ', 'from ')):
            return True
        
        return False
    
    def _is_prose_text(self, text: str) -> bool:
        """Check if text appears to be prose (sentences) rather than table data"""
        # Prose typically has:
        # - Multiple words with spaces
        # - Sentence-ending punctuation
        # - Connective words
        
        # Short text is unlikely to be prose
        if len(text) < 20:
            return False
        
        # Check for sentence patterns
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        word_count = len(text.split())
        
        # Prose typically has sentences and multiple words
        if sentence_endings > 0 and word_count > 5:
            # Check for common prose connectives
            connectives = ['and', 'or', 'but', 'which', 'that', 'this', 'these', 'those', 
                          'with', 'from', 'have', 'has', 'been', 'were', 'was', 'are', 'is']
            text_lower = text.lower()
            connective_count = sum(1 for word in connectives if f' {word} ' in f' {text_lower} ')
            
            # If has sentence endings and connectives, likely prose
            if connective_count >= 2:
                return True
        
        return False
    
    def _contains_number(self, text: str) -> bool:
        """Check if text contains numeric values"""
        # Remove common punctuation and check for numbers
        import re
        # Match integers, decimals, percentages, currency values, scientific notation
        number_patterns = [
            r'\d+\.?\d*',  # Basic numbers
            r'\$[\d,]+\.?\d*',  # Currency
            r'\d+%',  # Percentages
            r'\d{1,3}(,\d{3})*(\.\d+)?',  # Formatted numbers with commas
            r'\d+\.\d+[eE][+-]?\d+',  # Scientific notation
            r'^[+-]?\d*\.?\d+$'  # Just a number (entire cell)
        ]
    
    def _is_header_row(self, first_row, second_row) -> bool:
        """Check if first row looks like table headers"""
        if not first_row:
            return False
        
        header_indicators = 0
        total_cells = len([cell for cell in first_row if str(cell).strip()])
        
        if total_cells == 0:
            return False
        
        for cell in first_row:
            cell_str = str(cell).strip() if cell else ""
            
            # Headers are typically:
            # - Short to medium length (less than 100 chars)
            # - Title case or all caps
            # - No sentence-ending punctuation (except colon)
            # - Single line or short multi-line
            # - Often contain units in parentheses
            
            if cell_str:
                # Length check
                if len(cell_str) < 100:
                    header_indicators += 1
                
                # Case check (title case, all caps, or mixed with special chars)
                if cell_str.isupper() or cell_str.istitle() or any(c.isupper() for c in cell_str):
                    header_indicators += 1
                
                # No sentence-ending period
                if not cell_str.endswith('.'):
                    header_indicators += 1
                
                # Check for unit patterns like "(mm)", "[kg]", etc.
                if '(' in cell_str or '[' in cell_str:
                    header_indicators += 1
                
                # Short content (not paragraph-like)
                if cell_str.count(' ') < 10:  # Less than 10 spaces suggests short content
                    header_indicators += 1
        
        # Compare with second row if available
        if second_row:
            # Headers usually don't contain numbers, but data rows do
            first_row_numeric = sum(1 for cell in first_row if self._contains_number(str(cell)))
            second_row_numeric = sum(1 for cell in second_row if self._contains_number(str(cell)))
            
            # Strong indicator: first row has fewer numbers than second row
            if first_row_numeric < second_row_numeric:
                header_indicators += total_cells * 2  # Weight this heavily
        
        # Require average of at least 2 indicators per non-empty cell
        return header_indicators >= total_cells * 2
    
    def _validate_table_enhanced(self, table: Table) -> bool:
        """Enhanced validation for extracted table objects"""
        if not table.headers or not table.rows:
            return False
        
        # Check minimum dimensions
        if len(table.rows) < self.MIN_TABLE_ROWS:
            logger.debug(f"Table {table.id} rejected: Too few rows ({len(table.rows)})")
            return False
            
        num_cols = len(table.headers[0]) if table.headers and table.headers[0] else 0
        if num_cols < self.MIN_TABLE_COLS:
            logger.debug(f"Table {table.id} rejected: Too few columns ({num_cols})")
            return False
        
        # Check content quality
        total_cells = num_cols * len(table.rows)
        content_cells = 0
        numeric_cells = 0
        
        for row in table.rows:
            for cell in row:
                cell_str = str(cell).strip() if cell else ""
                if cell_str:
                    content_cells += 1
                    if self._contains_number(cell_str):
                        numeric_cells += 1
        
        if total_cells > 0:
            content_density = content_cells / total_cells
            if content_density < self.MIN_CONTENT_DENSITY:
                logger.debug(f"Table {table.id} rejected: Low content density ({content_density:.2f})")
                return False
            
            # Require some numeric content for data tables
            numeric_ratio = numeric_cells / content_cells if content_cells > 0 else 0
            if numeric_ratio < 0.05 and len(table.rows) > 2:  # Less than 5% numeric and more than 2 rows
                # Check if it's a valid text-only table (like a comparison table)
                if not self._is_valid_text_table(table):
                    logger.debug(f"Table {table.id} rejected: No numeric data and not a valid text table")
                    return False
        
        return True
    
    def _is_valid_text_table(self, table: Table) -> bool:
        """Check if a text-only table is valid (e.g., comparison tables, feature matrices)"""
        # Text tables should have:
        # - Consistent short entries
        # - Clear headers
        # - Structured content
        
        if not table.headers or not table.headers[0]:
            return False
        
        # Check if entries are reasonably short and consistent
        all_lengths = []
        for row in table.rows:
            for cell in row:
                cell_str = str(cell).strip() if cell else ""
                if cell_str:
                    all_lengths.append(len(cell_str))
        
        if not all_lengths:
            return False
        
        # Check if most cells are reasonably short (not paragraphs)
        avg_length = sum(all_lengths) / len(all_lengths)
        max_length = max(all_lengths)
        
        # Text tables should have short, consistent entries
        if avg_length > 100 or max_length > 300:
            return False
        
        return True
    
    def _is_valid_table_data(self, table_data) -> bool:
        """Legacy validation method - redirects to enhanced version"""
        return self._is_valid_table_data_enhanced(table_data)
    
    def _validate_table(self, table: Table) -> bool:
        """Legacy validation method - redirects to enhanced version"""
        return self._validate_table_enhanced(table)
    
    # ... rest of the methods remain the same ...
    
    def _extract_caption_near_table(self, page, table_data, page_num: int, table_idx: int) -> str:
        """Extract caption text near the table"""
        try:
            caption_text = ""
            
            # Get table position information from PDFPlumber
            if hasattr(page, 'find_tables'):
                # Try to find the specific table to get its position
                tables = page.find_tables()
                if table_idx < len(tables):
                    table_bbox = tables[table_idx].bbox
                    if table_bbox:
                        # Extract caption below the table (most common)
                        caption_text = self._search_caption_below_table(page, table_bbox)
                        
                        # If no caption below, search above
                        if not caption_text.strip():
                            caption_text = self._search_caption_above_table(page, table_bbox)
            
            # Fallback: search for text patterns that look like table captions
            if not caption_text.strip():
                caption_text = self._search_table_caption_patterns(page, page_num, table_idx)
            
            return caption_text.strip()
            
        except Exception as e:
            logger.warning(f"Caption extraction failed for table {page_num}.{table_idx}: {e}")
            return ""
    
    def _search_caption_below_table(self, page, table_bbox) -> str:
        """Search for caption text below the table"""
        try:
            caption_text = ""
            
            # Search for text below the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is below table (within reasonable distance)
                if (word_top > table_bbox[3] and word_top < table_bbox[3] + 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Below table caption search failed: {e}")
            return ""
    
    def _search_caption_above_table(self, page, table_bbox) -> str:
        """Search for caption text above the table"""
        try:
            caption_text = ""
            
            # Search for text above the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is above table (within reasonable distance)
                if (word_bottom < table_bbox[1] and word_bottom > table_bbox[1] - 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Above table caption search failed: {e}")
            return ""
    
    def _search_table_caption_patterns(self, page, page_num: int, table_idx: int) -> str:
        """Search for table caption patterns in text"""
        try:
            caption_text = ""
            
            # Look for common table caption patterns
            caption_patterns = [
                f"Table {page_num}.{table_idx}",
                f"Table {page_num}.{table_idx + 1}",  # Sometimes 1-indexed
                f"TABLE {page_num}.{table_idx}",
                f"TABLE {page_num}.{table_idx + 1}",
                f"Table {page_num}",
                f"TABLE {page_num}"
            ]
            
            # Extract all text from the page
            page_text = page.extract_text()
            if page_text:
                # Look for caption patterns
                for pattern in caption_patterns:
                    if pattern in page_text:
                        # Extract text around the pattern
                        pattern_index = page_text.find(pattern)
                        start = max(0, pattern_index - 200)
                        end = min(len(page_text), pattern_index + 200)
                        caption_text = page_text[start:end].strip()
                        break
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Pattern-based caption search failed: {e}")
            return ""
    
    def _create_table_from_data(self, table_data, page_num: int, table_idx, method: str, page=None) -> Table:
        """Create Table object from extracted data"""
        # Convert table data to DataFrame for easier processing
        df = pd.DataFrame(table_data)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Extract headers and rows, ensuring no None values
        headers = []
        if len(df) > 0:
            headers = [str(cell) if cell is not None else "" for cell in df.iloc[0].tolist()]
        
        rows = []
        if len(df) > 1:
            rows = [[str(cell) if cell is not None else "" for cell in row] for row in df.iloc[1:].values.tolist()]
        
        # Create bounding box (approximate)
        bbox = BoundingBox(
            x1=0, y1=0, x2=100, y2=100, page=page_num, confidence=0.8
        )
        
        # Extract caption if page object is provided
        caption = None
        if page:
            # Convert table_idx to string for caption extraction
            idx_str = str(table_idx).split('_')[0] if isinstance(table_idx, str) else str(table_idx)
            caption = self._extract_caption_near_table(page, table_data, page_num, int(idx_str))
        
        # Create label with proper indexing
        table_label = f"Table {page_num}.{table_idx}" if isinstance(table_idx, str) else f"Table {page_num}.{table_idx}"
        
        return Table(
            id=f"{method}_page{page_num}_table{table_idx}",
            label=table_label,
            caption=caption,
            page=page_num,
            bbox=bbox,
            headers=[headers] if headers else [],
            rows=rows,
            extraction_method=method,
            csv_path=None,
            html=None,
            image_path=None
        )
    
    def _deduplicate_tables(self, tables: List[Table]) -> List[Table]:
        """Remove duplicate tables based on content similarity"""
        unique_tables = []
        seen_content = set()
        
        for table in tables:
            # Create a more detailed content hash including headers and first few data rows
            header_str = "_".join(table.headers[0][:5]) if table.headers and table.headers[0] else ""
            
            # Include first 2 rows of data in the signature
            data_sig = ""
            if table.rows:
                for row in table.rows[:2]:
                    row_str = "_".join(str(cell)[:20] for cell in row[:5])
                    data_sig += row_str
            
            content_str = f"p{table.page}_h{len(table.headers)}_r{len(table.rows)}_{header_str}_{data_sig}"
            
            if content_str not in seen_content:
                seen_content.add(content_str)
                unique_tables.append(table)
            else:
                logger.debug(f"Duplicate table removed: {table.label}")
        
        return unique_tables
    
    async def _process_and_store_table(self, table: Table, pdf_path: Path) -> Optional[Table]:
        """Process table and store it based on STORE_LOCALLY setting"""
        try:
            # Generate unique filename
            filename = f"{table.extraction_method}_page{table.page}_table{table.id.split('_')[-1]}"
            
            html_content = None
            cloudinary_csv_url = None
            
            if settings.store_locally:
                # Store locally if enabled
                csv_path = self.output_dir / f"{filename}.csv"
                try:
                    # Create DataFrame and save CSV
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    df.to_csv(csv_path, index=False)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    logger.info(f"Created CSV file: {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to create CSV for table {table.id}: {e}")
                    return None
                
                # Upload CSV to Cloudinary
                try:
                    if csv_path.exists():
                        cloudinary_csv_url = await cloudinary_service.upload_file(
                            str(csv_path), 
                            folder="tables/csv",
                            public_id=f"{filename}_data"
                        )
                        logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                    
                # Update table object with file paths
                table.csv_path = cloudinary_csv_url or str(csv_path)
                
                # Save table metadata to JSON
                json_path = self.output_dir / f"{filename}.json"
                try:
                    table_dict = {
                        "id": table.id,
                        "page": table.page,
                        "extraction_method": table.extraction_method,
                        "headers": table.headers,
                        "rows": table.rows,
                        "csv_path": table.csv_path,
                        "extraction_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(table_dict, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved table metadata: {json_path}")
                except Exception as e:
                    logger.error(f"Failed to save table metadata: {e}")
            else:
                # Only store in Cloudinary, not locally
                try:
                    # Create DataFrame in memory
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    # Convert DataFrame to CSV bytes for Cloudinary upload
                    csv_buffer = io.BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue()
                    
                    # Upload CSV bytes to Cloudinary
                    cloudinary_csv_url = await cloudinary_service.upload_bytes(
                        csv_bytes, 
                        folder="tables/csv",
                        public_id=f"{filename}_data",
                        file_extension=".csv"
                    )
                    logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                    
                    # Update table object with Cloudinary URL
                    table.csv_path = cloudinary_csv_url
                    
                except Exception as e:
                    logger.error(f"Failed to upload CSV to Cloudinary: {e}")
                    return None
            
            # Set HTML content
            table.html = html_content
            
            return table
            
        except Exception as e:
            logger.error(f"Failed to process and store table {table.id}: {e}")
            return None
        
        for pattern in number_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_header_row(self, first_row, second_row) -> bool:
        """Check if first row looks like table headers"""
        if not first_row:
            return False
        
        header_indicators = 0
        
        for cell in first_row:
            cell_str = str(cell).strip() if cell else ""
            
            # Headers are typically:
            # - Short (less than 50 chars)
            # - Title case or all caps
            # - No punctuation at end (except colon)
            # - Single line
            
            if cell_str:
                if len(cell_str) < 50:
                    header_indicators += 1
                if cell_str.isupper() or cell_str.istitle():
                    header_indicators += 1
                if not cell_str.endswith('.'):
                    header_indicators += 1
                if '\n' not in cell_str:
                    header_indicators += 1
        
        # Compare with second row if available
        if second_row:
            # Headers usually don't contain numbers, but data rows do
            first_row_numeric = sum(1 for cell in first_row if self._contains_number(str(cell)))
            second_row_numeric = sum(1 for cell in second_row if self._contains_number(str(cell)))
            
            if first_row_numeric < second_row_numeric:
                header_indicators += 2
        
        return header_indicators >= len(first_row) * 2  # Average of 2 indicators per cell
    
    def _validate_table_enhanced(self, table: Table) -> bool:
        """Enhanced validation for extracted table objects"""
        if not table.headers or not table.rows:
            return False
        
        # Check minimum dimensions
        if len(table.rows) < self.MIN_TABLE_ROWS:
            logger.debug(f"Table {table.id} rejected: Too few rows ({len(table.rows)})")
            return False
            
        num_cols = len(table.headers[0]) if table.headers and table.headers[0] else 0
        if num_cols < self.MIN_TABLE_COLS:
            logger.debug(f"Table {table.id} rejected: Too few columns ({num_cols})")
            return False
        
        # Check content quality
        total_cells = num_cols * len(table.rows)
        content_cells = 0
        numeric_cells = 0
        
        for row in table.rows:
            for cell in row:
                cell_str = str(cell).strip() if cell else ""
                if cell_str:
                    content_cells += 1
                    if self._contains_number(cell_str):
                        numeric_cells += 1
        
        if total_cells > 0:
            content_density = content_cells / total_cells
            if content_density < self.MIN_CONTENT_DENSITY:
                logger.debug(f"Table {table.id} rejected: Low content density ({content_density:.2f})")
                return False
            
            # Require some numeric content for data tables
            numeric_ratio = numeric_cells / content_cells if content_cells > 0 else 0
            if numeric_ratio < 0.05 and len(table.rows) > 2:  # Less than 5% numeric and more than 2 rows
                # Check if it's a valid text-only table (like a comparison table)
                if not self._is_valid_text_table(table):
                    logger.debug(f"Table {table.id} rejected: No numeric data and not a valid text table")
                    return False
        
        return True
    
    def _is_valid_text_table(self, table: Table) -> bool:
        """Check if a text-only table is valid (e.g., comparison tables, feature matrices)"""
        # Text tables should have:
        # - Consistent short entries
        # - Clear headers
        # - Structured content
        
        if not table.headers or not table.headers[0]:
            return False
        
        # Check if entries are reasonably short and consistent
        all_lengths = []
        for row in table.rows:
            for cell in row:
                cell_str = str(cell).strip() if cell else ""
                if cell_str:
                    all_lengths.append(len(cell_str))
        
        if not all_lengths:
            return False
        
        # Check if most cells are reasonably short (not paragraphs)
        avg_length = sum(all_lengths) / len(all_lengths)
        max_length = max(all_lengths)
        
        # Text tables should have short, consistent entries
        if avg_length > 100 or max_length > 300:
                return False
        
        return True
    
    def _is_valid_table_data(self, table_data) -> bool:
        """Legacy validation method - redirects to enhanced version"""
        return self._is_valid_table_data_enhanced(table_data)
    
    def _validate_table(self, table: Table) -> bool:
        """Legacy validation method - redirects to enhanced version"""
        return self._validate_table_enhanced(table)
    
    # ... rest of the methods remain the same ...
    
    def _extract_caption_near_table(self, page, table_data, page_num: int, table_idx: int) -> str:
        """Extract caption text near the table"""
        try:
            caption_text = ""
            
            # Get table position information from PDFPlumber
            if hasattr(page, 'find_tables'):
                # Try to find the specific table to get its position
                tables = page.find_tables()
                if table_idx < len(tables):
                    table_bbox = tables[table_idx].bbox
                    if table_bbox:
                        # Extract caption below the table (most common)
                        caption_text = self._search_caption_below_table(page, table_bbox)
                        
                        # If no caption below, search above
                        if not caption_text.strip():
                            caption_text = self._search_caption_above_table(page, table_bbox)
            
            # Fallback: search for text patterns that look like table captions
            if not caption_text.strip():
                caption_text = self._search_table_caption_patterns(page, page_num, table_idx)
            
            return caption_text.strip()
            
        except Exception as e:
            logger.warning(f"Caption extraction failed for table {page_num}.{table_idx}: {e}")
            return ""
    
    def _search_caption_below_table(self, page, table_bbox) -> str:
        """Search for caption text below the table"""
        try:
            caption_text = ""
            
            # Search for text below the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is below table (within reasonable distance)
                if (word_top > table_bbox[3] and word_top < table_bbox[3] + 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Below table caption search failed: {e}")
            return ""
    
    def _search_caption_above_table(self, page, table_bbox) -> str:
        """Search for caption text above the table"""
        try:
            caption_text = ""
            
            # Search for text above the table
            for word in page.extract_words():
                word_x0, word_top, word_x1, word_bottom = word['x0'], word['top'], word['x1'], word['bottom']
                
                # Check if word is above table (within reasonable distance)
                if (word_bottom < table_bbox[1] and word_bottom > table_bbox[1] - 150 and
                    abs(word_x0 - (table_bbox[0] + table_bbox[2]) / 2) < 100):
                    caption_text += word['text'] + " "
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Above table caption search failed: {e}")
            return ""
    
    def _search_table_caption_patterns(self, page, page_num: int, table_idx: int) -> str:
        """Search for table caption patterns in text"""
        try:
            caption_text = ""
            
            # Look for common table caption patterns
            caption_patterns = [
                f"Table {page_num}.{table_idx}",
                f"Table {page_num}.{table_idx + 1}",  # Sometimes 1-indexed
                f"TABLE {page_num}.{table_idx}",
                f"TABLE {page_num}.{table_idx + 1}",
                f"Table {page_num}",
                f"TABLE {page_num}"
            ]
            
            # Extract all text from the page
            page_text = page.extract_text()
            if page_text:
                # Look for caption patterns
                for pattern in caption_patterns:
                    if pattern in page_text:
                        # Extract text around the pattern
                        pattern_index = page_text.find(pattern)
                        start = max(0, pattern_index - 200)
                        end = min(len(page_text), pattern_index + 200)
                        caption_text = page_text[start:end].strip()
                        break
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Pattern-based caption search failed: {e}")
            return ""
    
    def _create_table_from_data(self, table_data, page_num: int, table_idx, method: str, page=None) -> Table:
        """Create Table object from extracted data"""
        # Convert table data to DataFrame for easier processing
        df = pd.DataFrame(table_data)
        
        # Clean up the DataFrame
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Extract headers and rows, ensuring no None values
        headers = []
        if len(df) > 0:
            headers = [str(cell) if cell is not None else "" for cell in df.iloc[0].tolist()]
        
        rows = []
        if len(df) > 1:
            rows = [[str(cell) if cell is not None else "" for cell in row] for row in df.iloc[1:].values.tolist()]
        
        # Create bounding box (approximate)
        bbox = BoundingBox(
            x1=0, y1=0, x2=100, y2=100, page=page_num, confidence=0.8
        )
        
        # Extract caption if page object is provided
        caption = None
        if page:
            # Convert table_idx to string for caption extraction
            idx_str = str(table_idx).split('_')[0] if isinstance(table_idx, str) else str(table_idx)
            caption = self._extract_caption_near_table(page, table_data, page_num, int(idx_str))
        
        # Create label with proper indexing
        table_label = f"Table {page_num}.{table_idx}" if isinstance(table_idx, str) else f"Table {page_num}.{table_idx}"
        
        return Table(
            id=f"{method}_page{page_num}_table{table_idx}",
            label=table_label,
            caption=caption,
            page=page_num,
            bbox=bbox,
            headers=[headers] if headers else [],
            rows=rows,
            extraction_method=method,
            csv_path=None,
            html=None,
            image_path=None
        )
    
    def _deduplicate_tables(self, tables: List[Table]) -> List[Table]:
        """Remove duplicate tables based on content similarity"""
        unique_tables = []
        seen_content = set()
        
        for table in tables:
            # Create a more detailed content hash
            header_str = "_".join(table.headers[0]) if table.headers and table.headers[0] else ""
            content_str = f"{table.page}_{len(table.headers)}_{len(table.rows)}_{header_str[:50]}"
            
            if content_str not in seen_content:
                seen_content.add(content_str)
                unique_tables.append(table)
            else:
                logger.debug(f"Duplicate table removed: {table.label}")
        
        return unique_tables
    
    async def _process_and_store_table(self, table: Table, pdf_path: Path) -> Optional[Table]:
        """Process table and store it based on STORE_LOCALLY setting"""
        try:
            # Generate unique filename
            filename = f"{table.extraction_method}_page{table.page}_table{table.id.split('_')[-1]}"
            
            html_content = None
            cloudinary_csv_url = None
            
            if settings.store_locally:
                # Store locally if enabled
                csv_path = self.output_dir / f"{filename}.csv"
                try:
                    # Create DataFrame and save CSV
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    df.to_csv(csv_path, index=False)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    logger.info(f"Created CSV file: {csv_path}")
                except Exception as e:
                    logger.error(f"Failed to create CSV for table {table.id}: {e}")
                    return None
                
                # Upload CSV to Cloudinary
                try:
                    if csv_path.exists():
                        cloudinary_csv_url = await cloudinary_service.upload_file(
                            str(csv_path), 
                            folder="tables/csv",
                            public_id=f"{filename}_data"
                        )
                        logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                except Exception as e:
                    logger.warning(f"Failed to upload CSV to Cloudinary: {e}")
                    
                # Update table object with file paths
                table.csv_path = cloudinary_csv_url or str(csv_path)
                
                # Save table metadata to JSON
                json_path = self.output_dir / f"{filename}.json"
                try:
                    table_dict = {
                        "id": table.id,
                        "page": table.page,
                        "extraction_method": table.extraction_method,
                        "headers": table.headers,
                        "rows": table.rows,
                        "csv_path": table.csv_path,
                        "extraction_timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(table_dict, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"Saved table metadata: {json_path}")
                except Exception as e:
                    logger.error(f"Failed to save table metadata: {e}")
            else:
                # Only store in Cloudinary, not locally
                try:
                    # Create DataFrame in memory
                    df = pd.DataFrame(table.rows, columns=table.headers[0] if table.headers else None)
                    html_content = df.to_html(index=False, classes='table table-bordered')
                    
                    # Convert DataFrame to CSV bytes for Cloudinary upload
                    csv_buffer = io.BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_bytes = csv_buffer.getvalue()
                    
                    # Upload CSV bytes to Cloudinary
                    cloudinary_csv_url = await cloudinary_service.upload_bytes(
                        csv_bytes, 
                        folder="tables/csv",
                        public_id=f"{filename}_data",
                        file_extension=".csv"
                    )
                    logger.info(f"Uploaded CSV to Cloudinary: {cloudinary_csv_url}")
                    
                    # Update table object with Cloudinary URL
                    table.csv_path = cloudinary_csv_url
                    
                except Exception as e:
                    logger.error(f"Failed to upload CSV to Cloudinary: {e}")
                    return None
            
            # Set HTML content
            table.html = html_content
            
            return table
            
        except Exception as e:
            logger.error(f"Failed to process and store table {table.id}: {e}")
            return None