#!/usr/bin/env python3
"""
Local Storage Service for ScholarAI Extractor

This service handles local storage of all extracted content when STORE_LOCALLY=True.
It organizes content into structured folders and saves figures, tables, JSON data,
and text content in an organized manner.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from app.config import settings
from app.models.schemas import ExtractionResult, Figure, Table, Section


class LocalStorageService:
    """
    Service for storing extracted content locally in organized folder structure
    """
    
    def __init__(self):
        self.base_dir = Path("paper")
        self.figures_dir = self.base_dir / "figures"
        self.tables_dir = self.base_dir / "tables"
        self.json_dir = self.base_dir / "json"
        self.text_dir = self.base_dir / "text"
        self.sections_dir = self.text_dir / "sections"
        self.subsections_dir = self.text_dir / "subsections"
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all necessary directories"""
        directories = [
            self.figures_dir,
            self.tables_dir,
            self.json_dir,
            self.text_dir,
            self.sections_dir,
            self.subsections_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    async def store_extraction_result(self, result: ExtractionResult, job_id: str, paper_id: str) -> Dict[str, str]:
        """
        Store complete extraction result locally
        
        Args:
            result: The extraction result to store
            job_id: Job identifier
            paper_id: Paper identifier
            
        Returns:
            Dict with paths to stored content
        """
        if not settings.store_locally:
            logger.debug("Local storage disabled, skipping local storage")
            return {}
        
        try:
            stored_paths = {}
            
            # Use only paper_id for folder naming (no timestamp to avoid duplicates)
            extraction_dir = paper_id
            
            # Store figures
            if result.figures:
                figures_path = await self._store_figures(result.figures, extraction_dir)
                stored_paths['figures'] = figures_path
            
            # Store tables
            if result.tables:
                tables_path = await self._store_tables(result.tables, extraction_dir)
                stored_paths['tables'] = tables_path
            
            # Store JSON result
            json_path = await self._store_json_result(result, extraction_dir)
            stored_paths['json'] = json_path
            
            # Store text content (sections and subsections)
            if result.sections:
                text_path = await self._store_text_content(result.sections, extraction_dir)
                stored_paths['text'] = text_path
            
            logger.info(f"Stored extraction result locally for job {job_id} in {extraction_dir}")
            return stored_paths
            
        except Exception as e:
            logger.error(f"Failed to store extraction result locally: {e}")
            return {}
    
    async def _store_figures(self, figures: List[Figure], extraction_dir: str) -> str:
        """Store figures in organized folder structure"""
        figures_extraction_dir = self.figures_dir / extraction_dir
        figures_extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # Group figures by extraction method
        method_figures = {}
        for figure in figures:
            method = figure.extraction_method or 'unknown'
            if method not in method_figures:
                method_figures[method] = []
            method_figures[method].append(figure)
        
        # Store figures by method
        for method, method_figs in method_figures.items():
            method_dir = figures_extraction_dir / method
            method_dir.mkdir(exist_ok=True)
            
            for idx, figure in enumerate(method_figs):
                # Create figure metadata file
                figure_metadata = {
                    'label': figure.label,
                    'caption': figure.caption,
                    'page': figure.page,
                    'bbox': {
                        'x1': figure.bbox.x1,
                        'y1': figure.bbox.y1,
                        'x2': figure.bbox.x2,
                        'y2': figure.bbox.y2
                    },
                    'type': figure.type,
                    'confidence': figure.confidence,
                    'extraction_method': figure.extraction_method,
                    'image_path': figure.image_path,
                    'ocr_text': figure.ocr_text,
                    'ocr_confidence': figure.ocr_confidence,
                    'references': figure.references
                }
                
                metadata_file = method_dir / f"figure_{idx+1}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(figure_metadata, f, indent=2, ensure_ascii=False)
        
        return str(figures_extraction_dir)
    
    async def _store_tables(self, tables: List[Table], extraction_dir: str) -> str:
        """Store tables in organized folder structure"""
        tables_extraction_dir = self.tables_dir / extraction_dir
        tables_extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # Group tables by extraction method
        method_tables = {}
        for table in tables:
            method = table.extraction_method or 'unknown'
            if method not in method_tables:
                method_tables[method] = []
            method_tables[method].append(table)
        
        # Store tables by method
        for method, method_tabs in method_tables.items():
            method_dir = tables_extraction_dir / method
            method_dir.mkdir(exist_ok=True)
            
            for idx, table in enumerate(method_tabs):
                # Create table data file
                table_data = {
                    'headers': table.headers,
                    'rows': table.rows,
                    'page': table.page,
                    'bbox': {
                        'x1': table.bbox.x1,
                        'y1': table.bbox.y1,
                        'x2': table.bbox.x2,
                        'y2': table.bbox.y2
                    },
                    'confidence': getattr(table.bbox, 'confidence', None),
                    'extraction_method': table.extraction_method,
                    'image_path': table.image_path,
                    'html': table.html,
                    'structure': table.structure
                }
                
                data_file = method_dir / f"table_{idx+1}_data.json"
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(table_data, f, indent=2, ensure_ascii=False)
        
        return str(tables_extraction_dir)
    
    async def _store_json_result(self, result: ExtractionResult, extraction_dir: str) -> str:
        """Store complete JSON result"""
        json_extraction_dir = self.json_dir / extraction_dir
        json_extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert result to dict
        if hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
        else:
            result_dict = result
        
        # Store complete result
        result_file = json_extraction_dir / "extraction_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # Store individual components
        components_dir = json_extraction_dir / "components"
        components_dir.mkdir(exist_ok=True)
        
        # Store sections
        if result.sections:
            sections_file = components_dir / "sections.json"
            with open(sections_file, 'w', encoding='utf-8') as f:
                json.dump([section.model_dump() if hasattr(section, 'model_dump') else section.dict() for section in result.sections], f, indent=2, ensure_ascii=False, default=str)
        
        # Store figures
        if result.figures:
            figures_file = components_dir / "figures.json"
            with open(figures_file, 'w', encoding='utf-8') as f:
                json.dump([figure.model_dump() if hasattr(figure, 'model_dump') else figure.dict() for figure in result.figures], f, indent=2, ensure_ascii=False, default=str)
        
        # Store tables
        if result.tables:
            tables_file = components_dir / "tables.json"
            with open(tables_file, 'w', encoding='utf-8') as f:
                json.dump([table.model_dump() if hasattr(table, 'model_dump') else table.dict() for table in result.tables], f, indent=2, ensure_ascii=False, default=str)
        
        # Store equations
        if result.equations:
            equations_file = components_dir / "equations.json"
            with open(equations_file, 'w', encoding='utf-8') as f:
                json.dump([eq.model_dump() if hasattr(eq, 'model_dump') else eq.dict() for eq in result.equations], f, indent=2, ensure_ascii=False, default=str)
        
        # Store code blocks
        if result.code_blocks:
            code_file = components_dir / "code_blocks.json"
            with open(code_file, 'w', encoding='utf-8') as f:
                json.dump([code.model_dump() if hasattr(code, 'model_dump') else code.dict() for code in result.code_blocks], f, indent=2, ensure_ascii=False, default=str)
        
        # Store references
        if result.references:
            references_file = components_dir / "references.json"
            with open(references_file, 'w', encoding='utf-8') as f:
                json.dump([ref.model_dump() if hasattr(ref, 'model_dump') else ref.dict() for ref in result.references], f, indent=2, ensure_ascii=False, default=str)
        
        # Store entities
        if result.entities:
            entities_file = components_dir / "entities.json"
            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump([entity.model_dump() if hasattr(entity, 'model_dump') else entity.dict() for entity in result.entities], f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_extraction_dir)
    
    async def _store_text_content(self, sections: List[Section], extraction_dir: str) -> str:
        """Store text content in organized folder structure"""
        text_extraction_dir = self.text_dir / extraction_dir
        text_extraction_dir.mkdir(parents=True, exist_ok=True)
        
        # Store sections
        sections_dir = text_extraction_dir / "sections"
        sections_dir.mkdir(exist_ok=True)
        
        for idx, section in enumerate(sections):
            section_file = sections_dir / f"section_{idx+1}_{section.id or 'unnamed'}.txt"
            
            # Create section content
            section_content = f"Section: {section.title or 'Untitled'}\n"
            section_content += f"Level: {section.level}\n"
            section_content += f"Page: {section.page_start}-{section.page_end}\n"
            section_content += "=" * 50 + "\n\n"
            
            # Add paragraphs
            for para in section.paragraphs:
                # Handle both Paragraph objects and dicts
                if hasattr(para, 'text'):
                    text = para.text
                elif isinstance(para, dict) and 'text' in para:
                    text = para['text']
                else:
                    text = str(para)
                section_content += f"{text}\n\n"
            
            with open(section_file, 'w', encoding='utf-8') as f:
                f.write(section_content)
            
            # Store subsections if any
            if section.subsections:
                subsections_dir = sections_dir / f"section_{idx+1}_subsections"
                subsections_dir.mkdir(exist_ok=True)
                
                for sub_idx, subsection in enumerate(section.subsections):
                    subsection_file = subsections_dir / f"subsection_{sub_idx+1}_{subsection.id or 'unnamed'}.txt"
                    
                    subsection_content = f"Subsection: {subsection.title or 'Untitled'}\n"
                    subsection_content += f"Level: {subsection.level}\n"
                    subsection_content += f"Page: {subsection.page_start}-{subsection.page_end}\n"
                    subsection_content += "-" * 40 + "\n\n"
                    
                    # Add paragraphs
                    for para in subsection.paragraphs:
                        # Handle both Paragraph objects and dicts
                        if hasattr(para, 'text'):
                            text = para.text
                        elif isinstance(para, dict) and 'text' in para:
                            text = para['text']
                        else:
                            text = str(para)
                        subsection_content += f"{text}\n\n"
                    
                    with open(subsection_file, 'w', encoding='utf-8') as f:
                        f.write(subsection_content)
        
        return str(text_extraction_dir)


# Global instance
local_storage_service = LocalStorageService()
