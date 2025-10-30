#!/usr/bin/env python3
"""
Tabula Table Extractor Service

This service uses Tabula-py to extract tables from PDF documents.
Tabula is good at extracting tables from various PDF formats.
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import re

from app.models.schemas import Table, BoundingBox
from app.services.cloudinary_service import cloudinary_service


class TabulaExtractor:
    """Tabula-based table extractor with enhanced validation"""
    
    def __init__(self):
        self.name = "tabula"
        self._initialized = False
        
        # Enhanced validation thresholds
        self.MIN_TABLE_ROWS = 2  # Minimum 2 rows (header + data)
        self.MIN_TABLE_COLS = 2  # Minimum 2 columns
        self.MIN_CONTENT_DENSITY = 0.4  # At least 40% cells with content
        self.MAX_CELL_TEXT_LENGTH = 300  # Maximum characters per cell (reduced from 500)
        self.MIN_NUMERIC_CELLS_RATIO = 0.05  # At least 5% numeric cells for data tables (reduced from 0.1)
    
    async def _ensure_initialized(self):
        """Ensure Tabula is available"""
        if not self._initialized:
            try:
                import tabula
                self._initialized = True
                logger.info("Tabula extractor initialized successfully")
            except ImportError as e:
                logger.error(f"Tabula not available: {e}")
                raise ImportError("Tabula not installed. Install with: pip install tabula-py")
    
    def _contains_number(self, text: str) -> bool:
        """Check if text contains numeric values"""
        # Match integers, decimals, percentages, currency values, scientific notation
        number_patterns = [
            r'\d+\.?\d*',  # Basic numbers
            r'\$[\d,]+\.?\d*',  # Currency
            r'\d+%',  # Percentages
            r'\d{1,3}(,\d{3})*(\.\d+)?',  # Formatted numbers with commas
            r'\d+\.\d+[eE][+-]?\d+',  # Scientific notation
            r'^[+-]?\d*\.?\d+$',  # Just a number (entire cell)
        ]
        
        for pattern in number_patterns:
            if re.search(pattern, str(text)):
                return True
        return False
    
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
    
    def _remove_caption_rows_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that are likely captions or references from DataFrame"""
        rows_to_keep = []
        
        for idx, row in df.iterrows():
            # Concatenate row content for checking
            row_text = " ".join(str(cell).strip() for cell in row if pd.notna(cell))
            
            # Skip if it's a caption or reference
            if self._is_caption_or_reference(row_text):
                continue
            
            # Skip rows that are mostly empty
            non_empty_cells = sum(1 for cell in row if pd.notna(cell) and str(cell).strip())
            if non_empty_cells < len(row) * 0.3:
                continue
            
            rows_to_keep.append(idx)
        
        if rows_to_keep:
            return df.loc[rows_to_keep]
        else:
            return pd.DataFrame()  # Return empty DataFrame if no valid rows
    
    def _is_valid_dataframe_table(self, df: pd.DataFrame) -> bool:
        """Enhanced validation to check if DataFrame represents a real table"""
        
        # Basic checks
        if df.empty or df.shape[0] < self.MIN_TABLE_ROWS:
            logger.debug(f"Tabula table rejected: Too few rows ({df.shape[0]})")
            return False
        
        if df.shape[1] < self.MIN_TABLE_COLS:
            logger.debug(f"Tabula table rejected: Too few columns ({df.shape[1]})")
            return False
        
        # Remove completely empty rows and columns for analysis
        df_clean = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df_clean.empty or df_clean.shape[0] < self.MIN_TABLE_ROWS:
            logger.debug("Tabula table rejected: Empty after cleaning")
            return False
        
        # Check for and remove caption/figure text rows
        df_clean = self._remove_caption_rows_from_df(df_clean)
        
        if df_clean.empty or df_clean.shape[0] < self.MIN_TABLE_ROWS:
            logger.debug("Tabula table rejected: Only captions/references found")
            return False
        
        # Analyze content
        total_cells = df_clean.shape[0] * df_clean.shape[1]
        content_cells = 0
        numeric_cells = 0
        very_long_cells = 0
        prose_like_cells = 0
        
        for _, row in df_clean.iterrows():
            for cell in row:
                cell_str = str(cell).strip() if pd.notna(cell) else ""
                
                if cell_str and cell_str.lower() != 'nan':
                    content_cells += 1
                    
                    # Check for numeric content
                    if self._contains_number(cell_str):
                        numeric_cells += 1
                    
                    # Check for very long text (likely paragraph)
                    if len(cell_str) > self.MAX_CELL_TEXT_LENGTH:
                        very_long_cells += 1
                    
                    # Check for prose-like text
                    if self._is_prose_text(cell_str):
                        prose_like_cells += 1
        
        # Content density check
        if total_cells > 0:
            content_density = content_cells / total_cells
            if content_density < self.MIN_CONTENT_DENSITY:
                logger.debug(f"Tabula table rejected: Low content density ({content_density:.2f})")
                return False
        
        # Reject if contains too much prose
        if prose_like_cells > total_cells * 0.3:
            logger.debug("Tabula table rejected: Too much prose text")
            return False
        
        # Reject if cells contain paragraph-like text
        if very_long_cells > total_cells * 0.15:
            logger.debug("Tabula table rejected: Contains paragraph-like text")
            return False
        
        # Column consistency check
        # Check if columns have consistent data types or patterns
        col_consistency_score = 0
        for col in df_clean.columns:
            col_data = df_clean[col].dropna()
            if len(col_data) > 0:
                # Check if column has consistent numeric or text pattern
                numeric_count = sum(1 for val in col_data if self._contains_number(str(val)))
                if numeric_count == len(col_data) or numeric_count == 0:
                    col_consistency_score += 1
        
        col_consistency_ratio = col_consistency_score / df_clean.shape[1] if df_clean.shape[1] > 0 else 0
        
        # Require either significant numeric data or strong column consistency with headers
        has_numeric_data = (numeric_cells / max(content_cells, 1)) > self.MIN_NUMERIC_CELLS_RATIO
        has_consistent_structure = col_consistency_ratio > 0.6
        
        # More flexible validation
        if has_consistent_structure and df_clean.shape[0] >= 3:
            # Good structure with enough rows - likely valid
            return True
        elif has_numeric_data:
            # Has enough numeric data - likely a data table
            return True
        elif numeric_cells >= 2 and col_consistency_ratio > 0.4:
            # Has some numeric cells and reasonable consistency
            return True
        else:
            logger.debug("Tabula table rejected: No numeric data and poor structure")
            return False
        
        # Additional check: Avoid single-column extractions
        if df_clean.shape[1] <= 1:
            logger.debug("Tabula table rejected: Single column")
            return False
        
        # Check for minimum meaningful content
        if content_cells < 4:
            logger.debug("Tabula table rejected: Too little content")
            return False
        
        # Check for reasonable header pattern
        if df_clean.shape[0] >= 2:
            first_row = df_clean.iloc[0]
            second_row = df_clean.iloc[1]
            
            # Headers typically don't have numbers, data rows do
            first_row_numeric = sum(1 for cell in first_row if self._contains_number(str(cell)))
            second_row_numeric = sum(1 for cell in second_row if self._contains_number(str(cell)))
            
            # If first row has significantly more numbers than expected for headers, 
            # and no clear numeric pattern in data, likely not a table
            if first_row_numeric > df_clean.shape[1] * 0.5 and numeric_cells < 5:
                logger.debug("Tabula table rejected: No clear header pattern")
                return False
        
        return True
    
    def _extract_caption_for_tabula_table(self, pdf_path: Path, page_num: int, table_idx: int) -> str:
        """Extract caption for Tabula tables using PDFPlumber for text extraction"""
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                if page_num <= len(pdf.pages):
                    page = pdf.pages[page_num - 1]
                    
                    # Search for table caption patterns
                    caption_text = self._search_table_caption_patterns(page, page_num, table_idx)
                    
                    return caption_text.strip()
                    
        except Exception as e:
            logger.debug(f"Caption extraction failed for Tabula table {page_num}.{table_idx}: {e}")
            return ""
        
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
                f"TABLE {page_num}",
                f"Table {table_idx + 1}",  # Sometimes just table number
                f"TABLE {table_idx + 1}"
            ]
            
            # Extract all text from the page
            page_text = page.extract_text()
            if page_text:
                # Look for caption patterns
                for pattern in caption_patterns:
                    if pattern in page_text:
                        # Extract text around the pattern
                        pattern_index = page_text.find(pattern)
                        start = max(0, pattern_index)
                        end = min(len(page_text), pattern_index + 300)
                        
                        # Extract the caption line
                        caption_candidate = page_text[start:end]
                        # Get just the first sentence/line as caption
                        caption_lines = caption_candidate.split('\n')
                        if caption_lines:
                            caption_text = caption_lines[0].strip()
                            # Add second line if it's continuation (doesn't start with Table)
                            if len(caption_lines) > 1 and not caption_lines[1].strip().startswith(('Table', 'TABLE', 'Figure')):
                                caption_text += " " + caption_lines[1].strip()
                        break
            
            return caption_text.strip()
            
        except Exception as e:
            logger.debug(f"Pattern-based caption search failed: {e}")
            return ""
    
    async def extract(self, pdf_path: Path) -> List[Table]:
        """Extract tables using Tabula with enhanced validation"""
        await self._ensure_initialized()
        
        try:
            import tabula
            
            logger.info(f"Extracting tables with Tabula from {pdf_path}")
            
            extracted_tables = []
            
            # Try both lattice and stream modes separately for better results
            for mode in ['lattice', 'stream']:
                try:
                    logger.debug(f"Trying Tabula with {mode} mode")
                    
                    # Extract tables using current mode
                    tables = tabula.read_pdf(
                        str(pdf_path),
                        pages='all',
                        multiple_tables=True,
                        guess=False,  # Don't guess table structure
                        lattice=(mode == 'lattice'),  # Use lattice for bordered tables
                        stream=(mode == 'stream'),    # Use stream for borderless tables
                        pandas_options={'header': None},  # Don't assume first row is header
                        encoding='utf-8',
                        silent=True  # Suppress Java output
                    )
                    
                    logger.debug(f"Tabula {mode} mode found {len(tables)} candidates")
                    
                    # Process each extracted table
                    for table_idx, df in enumerate(tables):
                        try:
                            # Validate DataFrame as table
                            if not isinstance(df, pd.DataFrame):
                                continue
                            
                            # Apply enhanced validation
                            if not self._is_valid_dataframe_table(df):
                                continue
                            
                            # Clean the DataFrame
                            df_clean = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            # Remove caption rows again after cleaning
                            df_clean = self._remove_caption_rows_from_df(df_clean)
                            
                            if df_clean.empty:
                                continue
                            
                            # Determine page number (Tabula doesn't provide exact page info)
                            # We'll estimate based on table index
                            estimated_page = (table_idx // 2) + 1  # Rough estimate
                            
                            # Extract headers and rows
                            headers = []
                            rows = []
                            
                            if not df_clean.empty:
                                # First row as headers
                                headers = df_clean.iloc[0].fillna('').astype(str).tolist()
                                
                                # Remaining rows as data
                                if df_clean.shape[0] > 1:
                                    rows_data = df_clean.iloc[1:].fillna('').astype(str)
                                    rows = rows_data.values.tolist()
                            
                            # Skip if no meaningful data
                            if not headers or not rows:
                                continue
                            
                            # Create bounding box (Tabula doesn't provide precise bbox)
                            bbox_obj = BoundingBox(
                                x1=50, y1=50, x2=550, y2=750,
                                page=estimated_page,
                                confidence=0.6  # Lower confidence since position is estimated
                            )
                            
                            # Try to extract caption
                            caption = self._extract_caption_for_tabula_table(
                                pdf_path, estimated_page, len(extracted_tables)
                            )
                            
                            # Create table object
                            table_obj = Table(
                                id=f"tabula_{mode}_p{estimated_page}_t{len(extracted_tables)}",
                                label=f"Table {estimated_page}.{len(extracted_tables) + 1}",
                                caption=caption,
                                page=estimated_page,
                                bbox=bbox_obj,
                                headers=[headers],
                                rows=rows,
                                csv_path=None,
                                extraction_method=f"tabula_{mode}",
                                image_path=None,
                                confidence=0.6
                            )
                            
                            extracted_tables.append(table_obj)
                            logger.info(f"Tabula {mode} extracted valid table: {table_obj.label}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to process Tabula table {table_idx}: {e}")
                            continue
                
                except UnicodeDecodeError:
                    # Try with different encoding if UTF-8 fails
                    logger.warning(f"UTF-8 encoding failed for {mode} mode, trying latin-1")
                    try:
                        tables = tabula.read_pdf(
                            str(pdf_path),
                            pages='all',
                            multiple_tables=True,
                            guess=False,
                            lattice=(mode == 'lattice'),
                            stream=(mode == 'stream'),
                            pandas_options={'header': None},
                            encoding='latin-1',
                            silent=True
                        )
                        
                        # Process tables with same validation as above
                        for table_idx, df in enumerate(tables):
                            if isinstance(df, pd.DataFrame) and self._is_valid_dataframe_table(df):
                                # Clean and process the DataFrame
                                df_clean = df.dropna(how='all').dropna(axis=1, how='all')
                                df_clean = self._remove_caption_rows_from_df(df_clean)
                                
                                if not df_clean.empty and df_clean.shape[0] > 1:
                                    estimated_page = (table_idx // 2) + 1
                                    headers = df_clean.iloc[0].fillna('').astype(str).tolist()
                                    rows = df_clean.iloc[1:].fillna('').astype(str).values.tolist()
                                    
                                    if headers and rows:
                                        table_obj = Table(
                                            id=f"tabula_{mode}_p{estimated_page}_t{len(extracted_tables)}",
                                            label=f"Table {estimated_page}.{len(extracted_tables) + 1}",
                                            caption=self._extract_caption_for_tabula_table(
                                                pdf_path, estimated_page, len(extracted_tables)
                                            ),
                                            page=estimated_page,
                                            bbox=BoundingBox(
                                                x1=50, y1=50, x2=550, y2=750,
                                                page=estimated_page,
                                                confidence=0.6
                                            ),
                                            headers=[headers],
                                            rows=rows,
                                            csv_path=None,
                                            extraction_method=f"tabula_{mode}",
                                            image_path=None,
                                            confidence=0.6
                                        )
                                        extracted_tables.append(table_obj)
                                
                    except Exception as e:
                        logger.error(f"Tabula {mode} extraction with latin-1 also failed: {e}")
                        
                except Exception as e:
                    logger.error(f"Tabula {mode} extraction failed: {e}")
            
            # Deduplicate tables based on content similarity
            unique_tables = self._deduplicate_tables(extracted_tables)
            
            logger.info(f"Tabula successfully extracted {len(unique_tables)} valid tables")
            return unique_tables
            
        except Exception as e:
            logger.error(f"Tabula extraction failed: {e}")
            return []
    
    def _deduplicate_tables(self, tables: List[Table]) -> List[Table]:
        """Remove duplicate tables based on content similarity"""
        if not tables:
            return []
        
        unique_tables = []
        seen_signatures = set()
        
        for table in tables:
            # Create a signature based on headers and first few rows
            header_sig = "_".join(table.headers[0][:5]) if table.headers and table.headers[0] else ""
            rows_sig = ""
            if table.rows:
                # Use first 2 rows for signature
                for row in table.rows[:2]:
                    rows_sig += "_".join(str(cell)[:20] for cell in row[:5])
            
            signature = f"p{table.page}_h{len(table.headers)}_r{len(table.rows)}_{header_sig}_{rows_sig}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_tables.append(table)
            else:
                logger.debug(f"Duplicate Tabula table removed: {table.label}")
        
        return unique_tables


# Global instance
tabula_extractor = TabulaExtractor()