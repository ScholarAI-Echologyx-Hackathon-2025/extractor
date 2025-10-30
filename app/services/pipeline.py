# services/pipeline.py
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback
from loguru import logger
import re
import numpy as np
from textstat import flesch_reading_ease

from app.models.schemas import (
    ExtractionResult, ExtractionStatus, ExtractionRequest,
    Metadata, Section, Figure, Table, CodeBlock, Equation, Reference, Entity, Paragraph
)
from app.models.enums import EntityType
from app.config import settings
from app.services.extractors.grobid_extractor import GROBIDExtractor
from app.services.extractors.figure_extractor import FigureExtractor
# from app.services.extractors.table_extractor import TableExtractor  # Disabled for efficiency
from app.services.extractors.ocr_math_extractor import OCRMathExtractor
from app.services.extractors.code_extractor import CodeExtractor
from app.services.local_storage_service import local_storage_service
from app.utils.exceptions import ExtractionError


class ExtractionPipeline:
    """
    Enhanced extraction pipeline with quality assurance and auto-correction
    Implements fallback strategies and error recovery
    """
    
    def __init__(self):
        self.extractors = {}
        self._initialize_extractors()
        
        # Quality thresholds
        self.QUALITY_THRESHOLDS = {
            'overall_score': 0.7,
            'text_coherence': 0.6,
            'table_false_positive_rate': 0.3,
            'figure_consistency': 0.5,
            'extraction_coverage': 0.6
        }
    
    def _initialize_extractors(self):
        """Initialize all available extractors"""
        try:
            self.extractors['grobid'] = GROBIDExtractor()
            logger.info("GROBID extractor initialized")
        except Exception as e:
            logger.warning(f"GROBID extractor not available: {e}")
        
        try:
            self.extractors['figure'] = FigureExtractor()
            logger.info("Enhanced figure extractor initialized")
        except Exception as e:
            logger.warning(f"Enhanced figure extractor initialization failed: {e}")
        
        # Table extraction disabled for efficiency
        # try:
        #     self.extractors['table'] = TableExtractor()
        #     logger.info("Enhanced table extractor initialized")
        # except Exception as e:
        #     logger.warning(f"Enhanced table extractor initialization failed: {e}")
        logger.info("Table extraction disabled for efficiency")
        
        try:
            self.extractors['ocr_math'] = OCRMathExtractor(output_dir=settings.paper_folder / "ocr_math")
            logger.info("OCR/Math extractor initialized")
        except Exception as e:
            logger.warning(f"OCR/Math extractor initialization failed: {e}")
        
        try:
            self.extractors['code'] = CodeExtractor()
            logger.info("Code extractor initialized")
        except Exception as e:
            logger.warning(f"Code extractor initialization failed: {e}")
    
    async def extract(self, pdf_path: Path, request: ExtractionRequest, skip_local_storage: bool = False) -> ExtractionResult:
        """
        Enhanced extraction pipeline with quality assurance
        """
        start_time = datetime.utcnow()
        
        # Get page count early
        page_count = self._get_pdf_page_count(pdf_path)
        
        # Initialize result
        result = ExtractionResult(
            pdf_path=str(pdf_path),
            pdf_hash=self._calculate_file_hash(pdf_path),
            status=ExtractionStatus.PROCESSING,
            extraction_methods=[],
            metadata=Metadata(title="Unknown", page_count=page_count)
        )
        
        try:
            # Phase 1: Content Extraction (parallel execution)
            extraction_tasks = await self._run_extraction_tasks(pdf_path, request)
            
            # Process extraction results
            await self._process_extraction_results(result, extraction_tasks)
            
            # Phase 2: Initial Quality Assessment
            initial_quality = await self._assess_initial_quality(result)
            logger.info(f"Initial quality assessment: {initial_quality}")
            
            # Phase 3: Quality-Based Auto-Correction
            if initial_quality['needs_correction']:
                result = await self._auto_correct_extraction(result, initial_quality)
                logger.info("Applied auto-corrections to improve quality")
            
            # Phase 4: Text Flow Reconstruction
            if result.sections:
                result.sections = await self._reconstruct_text_flow(result.sections)
                logger.info("Reconstructed text flow")
            
            # Phase 5: Cross-Reference Enhancement
            await self._enhance_cross_references(result)
            
            # Phase 6: Final Quality Validation
            final_quality = await self._comprehensive_quality_validation(result)
            result.quality_metrics = final_quality
            
            # Phase 7: Set Status Based on Quality
            result.status = self._determine_final_status(final_quality)
            
        except Exception as e:
            result.status = ExtractionStatus.FAILED
            result.errors = [str(e)]
            logger.error(f"Pipeline failed: {traceback.format_exc()}")
        
        # Calculate processing time and finalize
        result.processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Set extraction methods used
        result.extraction_methods = list(extraction_tasks.keys())
        
        # Calculate confidence scores
        result.confidence_scores = self._calculate_confidence_scores(result, extraction_tasks)
        
        # Calculate extraction coverage
        result.extraction_coverage = self._calculate_extraction_coverage(result)
        
        await self._save_result(result, skip_local_storage)
        
        return result
    
    async def _run_extraction_tasks(self, pdf_path: Path, request: ExtractionRequest) -> Dict[str, Any]:
        """Run extraction tasks in parallel with proper error handling"""
        tasks = {}
        
        # Create tasks based on request
        if request.extract_text:
            if 'grobid' in self.extractors:
                tasks['grobid'] = asyncio.create_task(
                    self._safe_extract('grobid', pdf_path)
                )
            else:
                # Fallback to simple text extraction
                tasks['text_fallback'] = asyncio.create_task(
                    self._extract_text_fallback(pdf_path)
                )
        
        if request.extract_figures and 'figure' in self.extractors:
            tasks['figures'] = asyncio.create_task(
                self._safe_extract('figure', pdf_path)
            )
        
        # Table extraction disabled for efficiency
        # if request.extract_tables and 'table' in self.extractors:
        #     tasks['tables'] = asyncio.create_task(
        #         self._safe_extract('table', pdf_path)
        #     )
        
        if request.extract_code and 'code' in self.extractors:
            tasks['code'] = asyncio.create_task(
                self._safe_extract('code', pdf_path)
            )
        
        if request.use_ocr and 'ocr_math' in self.extractors:
            tasks['ocr_math'] = asyncio.create_task(
                self._safe_extract('ocr_math', pdf_path)
            )
        
        # Wait for all tasks with timeout
        timeout = request.timeout or settings.extraction_timeout
        results = await self._wait_for_tasks(tasks, timeout)
        
        return results
    
    async def _safe_extract(self, extractor_name: str, pdf_path: Path) -> Dict[str, Any]:
        """Safely execute extraction with error handling"""
        try:
            extractor = self.extractors[extractor_name]
            result = await extractor.extract(pdf_path)
            return {'success': True, 'data': result, 'method': extractor_name}
        except Exception as e:
            logger.error(f"Extraction failed for {extractor_name}: {e}")
            return {'success': False, 'error': str(e), 'method': extractor_name}
    
    async def _wait_for_tasks(self, tasks: Dict, timeout: int) -> Dict[str, Any]:
        """Wait for tasks with proper timeout handling"""
        done, pending = await asyncio.wait(
            tasks.values(),
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Collect results
        results = {}
        for task_name, task in tasks.items():
            if task.done():
                try:
                    results[task_name] = task.result()
                except Exception as e:
                    results[task_name] = {
                        'success': False, 
                        'error': str(e), 
                        'method': task_name
                    }
            else:
                results[task_name] = {
                    'success': False, 
                    'error': 'Task timed out', 
                    'method': task_name
                }
        
        return results
    
    async def _process_extraction_results(self, result: ExtractionResult, extraction_tasks: Dict[str, Any]):
        """Process results from extraction tasks and populate the result object"""
        
        for task_name, task_result in extraction_tasks.items():
            if not task_result.get('success', False):
                logger.warning(f"Task {task_name} failed: {task_result.get('error', 'Unknown error')}")
                continue
            
            data = task_result.get('data', {})
            method = task_result.get('method', task_name)
            
            # Add method to extraction methods
            if method not in result.extraction_methods:
                result.extraction_methods.append(method)
            
            # Process based on task type
            if task_name == 'grobid' or task_name == 'text_fallback':
                # Text extraction results
                if 'metadata' in data:
                    # Preserve page_count if it's already set correctly
                    new_metadata = data['metadata']
                    if hasattr(result.metadata, 'page_count') and result.metadata.page_count > 0:
                        # Keep the existing page_count if it's valid
                        new_metadata.page_count = result.metadata.page_count
                    result.metadata = new_metadata
                if 'sections' in data:
                    result.sections.extend(data['sections'])
                if 'references' in data:
                    result.references.extend(data['references'])
                    
            elif task_name == 'figures':
                # Figure extraction results
                if isinstance(data, list):
                    result.figures.extend(data)
                elif isinstance(data, dict) and 'figures' in data:
                    result.figures.extend(data['figures'])
                    
            elif task_name == 'tables':
                # Table extraction results
                if isinstance(data, list):
                    result.tables.extend(data)
                elif isinstance(data, dict) and 'tables' in data:
                    result.tables.extend(data['tables'])
                    
            elif task_name == 'code':
                # Code extraction results
                if isinstance(data, list):
                    result.code_blocks.extend(data)
                elif isinstance(data, dict) and 'code_blocks' in data:
                    result.code_blocks.extend(data['code_blocks'])
                    
            elif task_name == 'ocr_math':
                # OCR/Math extraction results
                if isinstance(data, dict):
                    if 'equations' in data:
                        result.equations.extend(data['equations'])
                    if 'entities' in data:
                        result.entities.extend(data['entities'])
    
    # Legacy extraction methods for backward compatibility
    async def _extract_with_grobid(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract using GROBID with fallback"""
        extractor = self.extractors['grobid']
        return await extractor.extract(pdf_path)
    
    async def _extract_figures(self, pdf_path: Path) -> List[Figure]:
        """Extract figures"""
        extractor = self.extractors['figure']
        return await extractor.extract(pdf_path)
    
    # Table extraction disabled for efficiency
    # async def _extract_tables(self, pdf_path: Path) -> List[Table]:
    #     """Extract tables"""
    #     extractor = self.extractors['table']
    #     return await extractor.extract(pdf_path)
    
    async def _extract_code(self, pdf_path: Path) -> List[CodeBlock]:
        """Extract code blocks"""
        extractor = self.extractors['code']
        return await extractor.extract(pdf_path)
    
    async def _extract_ocr_math(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract using OCR and math detection"""
        extractor = self.extractors['ocr_math']
        return await extractor.extract(pdf_path)
    
    async def _extract_text_fallback(self, pdf_path: Path) -> Dict[str, Any]:
        """Fallback text extraction using PyMuPDF when GROBID is not available"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    # Split text into paragraphs for better structure
                    paragraphs = self._split_text_into_paragraphs(text.strip())
                    
                    # Create a simple section for each page
                    section = Section(
                        title=f"Page {page_num + 1}",
                        page_start=page_num + 1,
                        page_end=page_num + 1,
                        paragraphs=[Paragraph(text=para, page=page_num + 1) for para in paragraphs if para.strip()]
                    )
                    sections.append(section)
            
            # Get page count before closing the document
            page_count = len(doc)
            doc.close()
            
            # Create basic metadata
            metadata = Metadata(
                title="Extracted Document",
                page_count=page_count,
                language="en"
            )
            
            return {
                'success': True,
                'data': {
                    'metadata': metadata,
                    'sections': sections,
                    'references': []
                },
                'method': 'text_fallback'
            }
            
        except Exception as e:
            logger.error(f"Text fallback extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'text_fallback'
            }
    
    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs"""
        import re
        
        # Split by double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\s{3,}', text)
        
        # Clean up paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 10:  # Only keep substantial paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _enhance_with_ocr(self, result: ExtractionResult, ocr_text: List[Dict]):
        """Enhance extraction result with OCR text"""
        # If no sections were extracted, create from OCR
        if not result.sections and ocr_text:
            for page_data in ocr_text:
                section = Section(
                    title=f"Page {page_data['page']}",
                    page_start=page_data['page'],
                    page_end=page_data['page'],
                    paragraphs=[Paragraph(
                        text=page_data['text'],
                        page=page_data['page']
                    )]
                )
                result.sections.append(section)
    
    def _cross_reference_content(self, result: ExtractionResult):
        """Cross-reference figures, tables, and code with text"""
        # Build reference map
        figure_refs = {}
        table_refs = {}
        
        # Find references in text
        for section in result.sections:
            for para in section.paragraphs:
                text = para.get('text', '') if isinstance(para, dict) else getattr(para, 'text', str(para))
                
                # Find figure references
                fig_patterns = [
                    r'Figure\s+(\d+)',
                    r'Fig\.\s*(\d+)',
                    r'figure\s+(\d+)'
                ]
                for pattern in fig_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        fig_num = match.group(1)
                        if fig_num not in figure_refs:
                            figure_refs[fig_num] = []
                        figure_refs[fig_num].append(section.id)
                
                # Find table references
                table_patterns = [
                    r'Table\s+(\d+)',
                    r'table\s+(\d+)'
                ]
                for pattern in table_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        table_num = match.group(1)
                        if table_num not in table_refs:
                            table_refs[table_num] = []
                        table_refs[table_num].append(section.id)
        
        # Update figures with references
        for figure in result.figures:
            if figure.label:
                # Extract number from label
                match = re.search(r'\d+', figure.label)
                if match:
                    fig_num = match.group()
                    if fig_num in figure_refs:
                        figure.references = figure_refs[fig_num]
        
        # Update tables with references
        for table in result.tables:
            if table.label:
                match = re.search(r'\d+', table.label)
                if match:
                    table_num = match.group()
                    if table_num in table_refs:
                        table.references = table_refs[table_num]
    
    async def _extract_entities(self, result: ExtractionResult) -> List[Entity]:
        """Extract named entities from the document"""
        entities = []
        
        # Extract dataset mentions
        dataset_patterns = [
            r'(?:dataset|corpus|benchmark):\s*(\w+)',
            r'(\w+)\s+dataset',
            r'(?:MNIST|CIFAR|ImageNet|COCO|WikiText|GLUE|SQuAD)'
        ]
        
        # Extract code/tool mentions
        code_patterns = [
            r'(?:github\.com/[\w-]+/[\w-]+)',
            r'(?:implementation|code):\s*([\w-]+)',
            r'(?:PyTorch|TensorFlow|JAX|scikit-learn|NumPy|Pandas)'
        ]
        
        # Extract method mentions
        method_patterns = [
            r'(?:algorithm|method|approach):\s*([\w\s]+)',
            r'(?:BERT|GPT|ResNet|LSTM|GRU|CNN|RNN|Transformer)'
        ]
        
        # Search in sections
        for section in result.sections:
            for para in section.paragraphs:
                text = para.get('text', '') if isinstance(para, dict) else str(getattr(para, 'text', para))
                page = para.page if hasattr(para, 'page') else section.page_start
                
                # Find datasets
                for pattern in dataset_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entity = Entity(
                            type=EntityType.DATASET,
                            name=match.group(1) if match.lastindex else match.group(),
                            page=page,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                        )
                        entities.append(entity)
                
                # Find code repositories
                for pattern in code_patterns:
                    for match in re.finditer(pattern, text):
                        entity = Entity(
                            type=EntityType.CODE,
                            name=match.group(1) if match.lastindex else match.group(),
                            page=page,
                            context=text[max(0, match.start()-50):min(len(text), match.end()+50)]
                        )
                        entities.append(entity)
        
        # Deduplicate entities
        unique_entities = []
        seen = set()
        for entity in entities:
            key = (entity.type, entity.name.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _calculate_coverage(self, result: ExtractionResult) -> float:
        """Calculate extraction coverage percentage"""
        score = 0.0
        max_score = 100.0
        
        # Check metadata (10%)
        if result.metadata and result.metadata.title != "Unknown":
            score += 10
        
        # Check sections (30%)
        if result.sections:
            section_score = min(30, len(result.sections) * 3)
            score += section_score
        
        # Check figures (15%)
        if result.figures:
            figure_score = min(15, len(result.figures) * 3)
            score += figure_score
        
        # Check tables (15%)
        if result.tables:
            table_score = min(15, len(result.tables) * 5)
            score += table_score
        
        # Check code blocks (10%)
        if result.code_blocks:
            code_score = min(10, len(result.code_blocks) * 2)
            score += code_score
        
        # Check equations (10%)
        if result.equations:
            eq_score = min(10, len(result.equations) * 1)
            score += eq_score
        
        # Check references (10%)
        if result.references:
            ref_score = min(10, len(result.references) * 0.5)
            score += ref_score
        
        return min(100.0, score)
    
    def _calculate_confidence(self, result: ExtractionResult) -> Dict[str, float]:
        """Calculate confidence scores for different extraction types"""
        confidence = {}
        
        # Text confidence (based on extraction method)
        if 'grobid' in result.extraction_methods:
            confidence['text'] = 0.9
        elif 'nougat_tesseract' in result.extraction_methods:
            confidence['text'] = 0.7
        else:
            confidence['text'] = 0.5
        
        # Figure confidence
        if result.figures:
            avg_conf = sum(f.bbox.confidence for f in result.figures if f.bbox and f.bbox.confidence) 
            count = sum(1 for f in result.figures if f.bbox and f.bbox.confidence)
            confidence['figures'] = avg_conf / count if count > 0 else 0.8
        
        # Table confidence
        if result.tables:
            avg_conf = sum(t.bbox.confidence for t in result.tables if t.bbox and t.bbox.confidence)
            count = sum(1 for t in result.tables if t.bbox and t.bbox.confidence)
            confidence['tables'] = avg_conf / count if count > 0 else 0.8
        
        # Code confidence
        confidence['code'] = 0.85 if result.code_blocks else 0.0
        
        # Math confidence
        confidence['equations'] = 0.9 if 'nougat' in str(result.extraction_methods) else 0.7
        
        return confidence
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file"""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.warning(f"Failed to get page count for {pdf_path}: {e}")
            return 0
    
    async def _save_result(self, result: ExtractionResult, skip_local_storage: bool = False):
        """Save extraction result based on STORE_LOCALLY setting"""
        # Convert to dict
        result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
        
        if settings.store_locally:
            # Local storage mode: save JSON file and store locally
            output_path = settings.paper_folder / f"{Path(result.pdf_path).stem}_extraction.json"
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Extraction result saved to {output_path}")
            
            # Store locally if not skipped
            if not skip_local_storage:
                try:
                    # Generate a job_id and paper_id for local storage
                    job_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                    paper_id = Path(result.pdf_path).stem
                    
                    stored_paths = await local_storage_service.store_extraction_result(
                        result, job_id, paper_id
                    )
                    logger.info(f"Stored extraction result locally: {stored_paths}")
                except Exception as e:
                    logger.error(f"Failed to store extraction result locally: {e}")
        else:
            # Cloud-only mode: do not save or upload anything
            logger.info("STORE_LOCALLY=False: Skipping extraction result storage (no local files, no Cloudinary upload)")
            logger.debug(f"Extraction result for {Path(result.pdf_path).stem} completed but not stored")

    # Enhanced quality assessment and auto-correction methods
    async def _assess_initial_quality(self, result: ExtractionResult) -> Dict[str, Any]:
        """Assess initial extraction quality and identify issues"""
        quality_issues = []
        needs_correction = False
        
        # 1. Table false positive assessment
        if result.tables:
            false_positive_rate = await self._assess_table_false_positives(result.tables)
            if false_positive_rate > self.QUALITY_THRESHOLDS['table_false_positive_rate']:
                quality_issues.append({
                    'type': 'table_false_positives',
                    'severity': 'high',
                    'rate': false_positive_rate,
                    'affected_tables': self._identify_false_positive_tables(result.tables)
                })
                needs_correction = True
        
        # 2. Text coherence assessment
        if result.sections:
            coherence_score = self._assess_text_coherence(result.sections)
            if coherence_score < self.QUALITY_THRESHOLDS['text_coherence']:
                quality_issues.append({
                    'type': 'poor_text_coherence',
                    'severity': 'medium',
                    'score': coherence_score
                })
                needs_correction = True
        
        # 3. Figure-reference consistency
        if result.figures:
            consistency_score = self._assess_figure_consistency(result)
            if consistency_score < self.QUALITY_THRESHOLDS['figure_consistency']:
                quality_issues.append({
                    'type': 'figure_inconsistency',
                    'severity': 'low',
                    'score': consistency_score
                })
        
        return {
            'needs_correction': needs_correction,
            'issues': quality_issues,
            'overall_score': self._calculate_overall_quality_score(quality_issues)
        }
    
    async def _assess_table_false_positives(self, tables: List[Table]) -> float:
        """Assess rate of false positive tables"""
        if not tables:
            return 0.0
        
        false_positives = 0
        
        for table in tables:
            # Check if table content is actually prose text
            if self._is_table_actually_prose(table):
                false_positives += 1
        
        return false_positives / len(tables)
    
    def _is_table_actually_prose(self, table: Table) -> bool:
        """Determine if a table is actually prose text"""
        if not table.rows:
            return True
        
        # Combine all cell content
        all_text = []
        if table.headers:
            for header_row in table.headers:
                all_text.extend(str(cell) for cell in header_row)
        
        for row in table.rows:
            all_text.extend(str(cell) for cell in row)
        
        combined_text = ' '.join(all_text)
        
        if not combined_text.strip():
            return True
        
        # Multiple indicators of prose text
        indicators = 0
        
        # 1. High reading ease (typical of continuous prose)
        try:
            reading_ease = flesch_reading_ease(combined_text)
            if reading_ease > 60:  # Reasonably readable prose
                indicators += 1
        except:
            pass
        
        # 2. Complete sentences
        sentences = re.split(r'[.!?]+', combined_text)
        complete_sentences = sum(1 for s in sentences if len(s.strip().split()) > 4)
        if complete_sentences > len(table.rows):  # More complete sentences than table rows
            indicators += 1
        
        # 3. Low data diversity (mostly text, few numbers)
        numeric_cells = sum(1 for text in all_text 
                           if re.match(r'^\d+(\.\d+)?$', str(text).strip()))
        if numeric_cells / len(all_text) < 0.2:  # Less than 20% numeric
            indicators += 1
        
        # 4. Long continuous text spans
        long_text_cells = sum(1 for text in all_text 
                             if len(str(text).split()) > 8)
        if long_text_cells > len(table.rows) * 0.3:  # Many cells with long text
            indicators += 1
        
        # Consider it prose if multiple indicators are present
        return indicators >= 2
    
    def _identify_false_positive_tables(self, tables: List[Table]) -> List[str]:
        """Identify specific false positive table IDs"""
        false_positives = []
        for table in tables:
            if self._is_table_actually_prose(table):
                false_positives.append(table.label or f"table_page_{table.page}")
        return false_positives
    
    def _assess_text_coherence(self, sections: List[Section]) -> float:
        """Assess overall text coherence"""
        if not sections:
            return 0.0
        
        coherence_scores = []
        
        for section in sections:
            # Check paragraph flow
            if section.paragraphs:
                paragraph_texts = []
                for para in section.paragraphs:
                    if isinstance(para, dict):
                        paragraph_texts.append(para.get('text', ''))
                    else:
                        paragraph_texts.append(str(para.text if hasattr(para, 'text') else ''))
                
                section_text = ' '.join(paragraph_texts)
                
                if section_text.strip():
                    # Measure coherence indicators
                    coherence = self._calculate_text_coherence_score(section_text)
                    coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_text_coherence_score(self, text: str) -> float:
        """Calculate coherence score for text"""
        score = 0.0
        
        # 1. Sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
        
        if valid_sentences:
            score += 0.3
            
            # 2. Average sentence length (reasonable range)
            avg_length = np.mean([len(s.split()) for s in valid_sentences])
            if 8 <= avg_length <= 25:  # Reasonable academic sentence length
                score += 0.2
        
        # 3. Reading ease
        try:
            ease = flesch_reading_ease(text)
            if 30 <= ease <= 70:  # Academic text range
                score += 0.2
        except:
            score += 0.1  # Default if calculation fails
        
        # 4. Vocabulary diversity
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.4:  # Good vocabulary diversity
                score += 0.2
        
        # 5. Proper capitalization and punctuation
        if re.search(r'[A-Z][a-z]', text):  # Has proper capitalization
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_figure_consistency(self, result: ExtractionResult) -> float:
        """Assess consistency between figures and text references"""
        if not result.figures:
            return 1.0
        
        # Find figure references in text
        figure_refs = set()
        for section in result.sections:
            for para in section.paragraphs:
                text = para.text if hasattr(para, 'text') else str(para)
                refs = re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', text, re.IGNORECASE)
                figure_refs.update(refs)
        
        if not figure_refs:
            return 0.5  # No references found
        
        # Check how many figures have matching references
        matched_figures = 0
        for figure in result.figures:
            if figure.label:
                fig_num = re.search(r'\d+', figure.label)
                if fig_num and fig_num.group() in figure_refs:
                    matched_figures += 1
        
        return matched_figures / len(result.figures)
    
    def _calculate_overall_quality_score(self, quality_issues: List[Dict]) -> float:
        """Calculate overall quality score based on issues and content coverage"""
        # Start with a dynamic base score based on content extraction success
        base_score = 0.5  # Start lower and build up
        
        # Add points for successful extractions
        if hasattr(self, 'extraction_results') and self.extraction_results:
            successful_extractions = sum(1 for result in self.extraction_results.values() 
                                       if result.get('success', False))
            total_extractions = len(self.extraction_results)
            if total_extractions > 0:
                base_score += 0.3 * (successful_extractions / total_extractions)
        
        # Subtract points for issues
        for issue in quality_issues:
            if issue['severity'] == 'high':
                base_score -= 0.2
            elif issue['severity'] == 'medium':
                base_score -= 0.1
            elif issue['severity'] == 'low':
                base_score -= 0.05
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, base_score))
    
    async def _auto_correct_extraction(self, result: ExtractionResult, 
                                     quality_assessment: Dict[str, Any]) -> ExtractionResult:
        """Apply automatic corrections based on quality assessment"""
        corrected_result = result
        
        for issue in quality_assessment['issues']:
            if issue['type'] == 'table_false_positives' and issue['severity'] == 'high':
                corrected_result = await self._correct_table_false_positives(
                    corrected_result, issue
                )
            
            elif issue['type'] == 'poor_text_coherence' and issue['severity'] == 'medium':
                corrected_result = await self._correct_text_coherence_issues(
                    corrected_result
                )
        
        # Validate and correct page numbers
        corrected_result = await self._validate_page_numbers(corrected_result)
        
        # Improve section page number accuracy
        corrected_result = self._improve_section_page_accuracy(corrected_result)
        
        return corrected_result
    
    async def _correct_table_false_positives(self, result: ExtractionResult, 
                                           issue: Dict[str, Any]) -> ExtractionResult:
        """Remove false positive tables and convert to text"""
        false_positive_labels = issue['affected_tables']
        
        # Separate valid tables from false positives
        valid_tables = []
        false_positive_tables = []
        
        for table in result.tables:
            table_label = table.label or f"table_page_{table.page}"
            if table_label in false_positive_labels:
                false_positive_tables.append(table)
            else:
                valid_tables.append(table)
        
        # Convert false positive tables back to text
        for fp_table in false_positive_tables:
            text_content = self._convert_table_to_text(fp_table)
            
            # Add to appropriate section or create new section
            section = self._find_or_create_section_for_page(result.sections, fp_table.page)
            section.paragraphs.append({
                'text': text_content,
                'page': fp_table.page
            })
        
        result.tables = valid_tables
        logger.info(f"Removed {len(false_positive_tables)} false positive tables")
        
        return result
    
    def _convert_table_to_text(self, table: Table) -> str:
        """Convert table content back to readable text"""
        text_parts = []
        
        # Add headers as text
        if table.headers:
            for header_row in table.headers:
                header_text = ' '.join(str(cell) for cell in header_row if cell)
                if header_text.strip():
                    text_parts.append(header_text)
        
        # Add row content as text
        for row in table.rows:
            row_text = ' '.join(str(cell) for cell in row if cell)
            if row_text.strip():
                text_parts.append(row_text)
        
        return ' '.join(text_parts)
    
    def _find_or_create_section_for_page(self, sections: List[Section], 
                                       page: int) -> Section:
        """Find existing section for page or create new one"""
        # Look for existing section covering this page
        for section in sections:
            if section.page_start <= page <= section.page_end:
                return section
        
        # Create new section
        new_section = Section(
            title=f"Page {page}",
            page_start=page,
            page_end=page,
            paragraphs=[]
        )
        sections.append(new_section)
        return new_section
    
    async def _correct_text_coherence_issues(self, result: ExtractionResult) -> ExtractionResult:
        """Improve text coherence by reconstructing flow and fixing page numbers"""
        if not result.sections:
            return result
        
        # Sort sections by page order
        result.sections.sort(key=lambda s: s.page_start)
        
        # Merge fragmented sections on same page
        merged_sections = []
        current_section = None
        
        for section in result.sections:
            if current_section and current_section.page_end == section.page_start:
                # Merge with current section
                current_section.paragraphs.extend(section.paragraphs)
                current_section.page_end = section.page_end
                current_section.title = f"{current_section.title} (continued)"
            else:
                if current_section:
                    merged_sections.append(current_section)
                current_section = section
        
        if current_section:
            merged_sections.append(current_section)
        
        # Improve page number accuracy for each section
        for section in merged_sections:
            self._improve_section_page_numbers(section)
        
        result.sections = merged_sections
        return result
    
    def _improve_section_page_numbers(self, section: Section):
        """Improve page number accuracy for a section based on paragraph content"""
        if not section.paragraphs:
            return
        
        # Get all unique page numbers from paragraphs
        page_numbers = set()
        for para in section.paragraphs:
            if hasattr(para, 'page'):
                page_numbers.add(para.page)
            elif isinstance(para, dict) and 'page' in para:
                page_numbers.add(para['page'])
        
        if page_numbers:
            # Update section page range based on actual paragraph pages
            section.page_start = min(page_numbers)
            section.page_end = max(page_numbers)
            
            # If section spans multiple pages, check for logical breaks
            if section.page_end - section.page_start > 2:
                # Look for natural section breaks
                self._detect_section_boundaries(section)
    
    def _detect_section_boundaries(self, section: Section):
        """Detect natural section boundaries within a multi-page section"""
        # Group paragraphs by page
        page_paragraphs = {}
        for para in section.paragraphs:
            page = getattr(para, 'page', 1) if hasattr(para, 'page') else para.get('page', 1)
            if page not in page_paragraphs:
                page_paragraphs[page] = []
            page_paragraphs[page].append(para)
        
        # Look for pages with section-like content (headers, etc.)
        section_indicators = ['introduction', 'method', 'result', 'conclusion', 'discussion', 'abstract']
        
        for page_num, paras in page_paragraphs.items():
            page_text = ' '.join([getattr(p, 'text', '') if hasattr(p, 'text') else str(p) for p in paras])
            page_text_lower = page_text.lower()
            
            # Check if this page contains section indicators
            for indicator in section_indicators:
                if indicator in page_text_lower:
                    # This might be a new section, adjust page_end
                    if page_num > section.page_start:
                        section.page_end = page_num - 1
                    break
    
    async def _validate_page_numbers(self, result: ExtractionResult) -> ExtractionResult:
        """Validate and correct page numbers across all sections"""
        if not result.sections:
            return result
        
        # Get total page count from metadata
        total_pages = result.metadata.page_count if hasattr(result.metadata, 'page_count') else 0
        
        for section in result.sections:
            # Validate page_start and page_end
            if section.page_start < 1:
                section.page_start = 1
            
            if total_pages > 0 and section.page_end > total_pages:
                section.page_end = total_pages
            
            # Ensure page_end >= page_start
            if section.page_end < section.page_start:
                section.page_end = section.page_start
            
            # Validate paragraph page numbers
            for para in section.paragraphs:
                if hasattr(para, 'page'):
                    if para.page < 1:
                        para.page = 1
                    elif total_pages > 0 and para.page > total_pages:
                        para.page = total_pages
                elif isinstance(para, dict) and 'page' in para:
                    if para['page'] < 1:
                        para['page'] = 1
                    elif total_pages > 0 and para['page'] > total_pages:
                        para['page'] = total_pages
        
        return result
    
    def _improve_section_page_accuracy(self, result: ExtractionResult) -> ExtractionResult:
        """Improve section page number accuracy using paragraph content analysis"""
        if not result.sections:
            return result
        
        # Get total page count from metadata
        total_pages = result.metadata.page_count if hasattr(result.metadata, 'page_count') else 0
        
        for section in result.sections:
            if not section.paragraphs:
                continue
            
            # Collect all valid page numbers from paragraphs
            valid_page_numbers = []
            for para in section.paragraphs:
                if hasattr(para, 'page') and para.page > 0:
                    valid_page_numbers.append(para.page)
                elif isinstance(para, dict) and 'page' in para and para['page'] > 0:
                    valid_page_numbers.append(para['page'])
            
            if not valid_page_numbers:
                continue
            
            # Sort page numbers
            valid_page_numbers.sort()
            
            # Calculate new page range
            new_page_start = valid_page_numbers[0]
            new_page_end = valid_page_numbers[-1]
            
            # Validate the new range
            if new_page_end < new_page_start:
                new_page_end = new_page_start
            
            # Check if the new range is more reasonable than the current one
            current_span = section.page_end - section.page_start
            new_span = new_page_end - new_page_start
            
            # If new range is significantly smaller and more reasonable, use it
            if new_span < current_span and new_span <= 10:  # Most sections don't span more than 10 pages
                section.page_start = new_page_start
                section.page_end = new_page_end
                logger.info(f"Updated section '{section.title}' page range from {current_span} to {new_span} pages")
            
            # Additional validation: if section spans too many pages, look for natural breaks
            elif new_span > 10:
                # Try to find a more reasonable range by looking for content concentration
                page_counts = {}
                for page_num in valid_page_numbers:
                    page_counts[page_num] = page_counts.get(page_num, 0) + 1
                
                # Find the page with most content
                if page_counts:
                    most_content_page = max(page_counts, key=page_counts.get)
                    # Limit range to ±2 pages around the most content-rich page
                    section.page_start = max(1, most_content_page - 2)
                    section.page_end = min(total_pages, most_content_page + 2)
                    logger.info(f"Limited section '{section.title}' page range to reasonable span around page {most_content_page}")
        
        return result
    
    def _calculate_confidence_scores(self, result: ExtractionResult, extraction_tasks: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for each extraction method"""
        confidence_scores = {}
        
        # Base confidence scores for different methods
        method_confidence = {
            'grobid': 0.9,
            'text_fallback': 0.7,
            'transformer': 0.85,
            'pdfplumber': 0.8,

            'tabula': 0.7,

            'cv_contour': 0.6,

    
            'nougat_tesseract': 0.8,
            'ocr_math': 0.7
        }
        
        for method_name, task_result in extraction_tasks.items():
            if task_result.get('success', False):
                # Get base confidence for this method
                base_confidence = method_confidence.get(method_name, 0.5)
                
                # Adjust based on content quality
                content_quality = self._assess_content_quality(method_name, task_result)
                
                # Final confidence score
                confidence_scores[method_name] = base_confidence * content_quality
            else:
                confidence_scores[method_name] = 0.0
        
        return confidence_scores
    
    def _assess_content_quality(self, method_name: str, task_result: Dict[str, Any]) -> float:
        """Assess the quality of extracted content"""
        data = task_result.get('data', {})
        
        if method_name in ['grobid', 'text_fallback']:
            # Assess text quality
            sections = data.get('sections', [])
            if sections:
                # Check for substantial content
                total_text_length = sum(len(str(para.get('text', '') if isinstance(para, dict) else getattr(para, 'text', para))) 
                                      for section in sections
                                      for para in section.paragraphs)
                return min(1.0, total_text_length / 1000)  # Normalize by expected length
            return 0.5
        
        elif method_name in ['transformer', 'pdfplumber', 'tabula']:
            # Assess table quality
            tables = data if isinstance(data, list) else data.get('tables', [])
            if tables:
                # Check for tables with substantial content
                valid_tables = sum(1 for table in tables 
                                 if table.rows and len(table.rows) > 1)
                return valid_tables / len(tables) if tables else 0.0
            return 0.5
        
        elif method_name in ['cv_contour']:
            # Assess figure quality
            figures = data if isinstance(data, list) else data.get('figures', [])
            if figures:
                # Check for figures with captions or substantial content
                valid_figures = sum(1 for fig in figures 
                                  if fig.caption or fig.ocr_text)
                return valid_figures / len(figures) if figures else 0.0
            return 0.5
        
        return 0.5  # Default quality score
    
    def _calculate_extraction_coverage(self, result: ExtractionResult) -> float:
        """Calculate overall extraction coverage"""
        coverage_scores = []
        
        # Text coverage
        if result.sections:
            total_paragraphs = sum(len(section.paragraphs) for section in result.sections)
            if total_paragraphs > 0:
                coverage_scores.append(min(1.0, total_paragraphs / 50))  # Normalize by expected paragraphs
        
        # Table coverage
        if result.tables:
            coverage_scores.append(min(1.0, len(result.tables) / 10))  # Normalize by expected tables
        
        # Figure coverage
        if result.figures:
            coverage_scores.append(min(1.0, len(result.figures) / 15))  # Normalize by expected figures
        
        # Equation coverage
        if result.equations:
            coverage_scores.append(min(1.0, len(result.equations) / 20))  # Normalize by expected equations
        
        # Code coverage
        if result.code_blocks:
            coverage_scores.append(min(1.0, len(result.code_blocks) / 5))  # Normalize by expected code blocks
        
        # Reference coverage
        if result.references:
            coverage_scores.append(min(1.0, len(result.references) / 30))  # Normalize by expected references
        
        # Calculate overall coverage
        if coverage_scores:
            return sum(coverage_scores) / len(coverage_scores)
        else:
            return 0.0
    
    async def _reconstruct_text_flow(self, sections: List[Section]) -> List[Section]:
        """Reconstruct proper text reading order"""
        reconstructed_sections = []
        
        for section in sections:
            # Sort paragraphs by page order
            if section.paragraphs:
                sorted_paragraphs = sorted(
                    section.paragraphs,
                    key=lambda p: p.get('page', 0) if isinstance(p, dict) else getattr(p, 'page', 0)
                )
                
                # Merge paragraphs that should be continuous
                merged_paragraphs = self._merge_continuous_paragraphs(sorted_paragraphs)
                section.paragraphs = merged_paragraphs
            
            reconstructed_sections.append(section)
        
        return reconstructed_sections
    
    def _merge_continuous_paragraphs(self, paragraphs: List) -> List:
        """Merge paragraphs that should be continuous"""
        if not paragraphs:
            return paragraphs
        
        merged = []
        current_text = ""
        current_page = None
        
        for para in paragraphs:
            if isinstance(para, dict):
                text = para.get('text', '')
                page = para.get('page', 0)
            else:
                text = str(getattr(para, 'text', para))
                page = getattr(para, 'page', 0)
            
            text = text.strip()
            if not text:
                continue
            
            # Check if this paragraph continues from previous
            if current_text and self._should_merge_paragraphs(current_text, text):
                # Merge with current paragraph
                current_text += ' ' + text
            else:
                # Start new paragraph
                if current_text:
                    merged.append(Paragraph(
                        text=current_text,
                        page=current_page or page
                    ))
                current_text = text
                current_page = page
        
        # Add final paragraph
        if current_text:
            merged.append(Paragraph(
                text=current_text,
                page=current_page or 1
            ))
        
        return merged
    
    def _should_merge_paragraphs(self, text1: str, text2: str) -> bool:
        """Determine if two paragraphs should be merged"""
        # Don't merge if first paragraph ends with period
        if text1.rstrip().endswith('.'):
            return False
        
        # Don't merge if second paragraph starts with capital letter (likely new sentence)
        if text2.lstrip() and text2.lstrip()[0].isupper():
            return False
        
        # Merge if first paragraph seems incomplete
        if text1.rstrip().endswith((',', ';', ':', '-')):
            return True
        
        return False
    
    async def _enhance_cross_references(self, result: ExtractionResult):
        """Enhanced cross-referencing between content types"""
        # Build comprehensive reference map
        reference_map = self._build_reference_map(result)
        
        # Update figures with references
        for figure in result.figures:
            figure.references = reference_map.get('figures', {}).get(
                self._extract_number_from_label(figure.label), []
            )
        
        # Update tables with references
        for table in result.tables:
            table.references = reference_map.get('tables', {}).get(
                self._extract_number_from_label(table.label), []
            )
        
        # Update code blocks with references
        for code in result.code_blocks:
            if hasattr(code, 'references'):
                code.references = reference_map.get('code', {}).get(
                    self._extract_number_from_label(getattr(code, 'label', '')), []
                )
    
    def _build_reference_map(self, result: ExtractionResult) -> Dict[str, Dict[str, List[str]]]:
        """Build comprehensive map of content references"""
        reference_map = {
            'figures': {},
            'tables': {},
            'code': {},
            'equations': {}
        }
        
        # Search through all text for references
        all_text = []
        section_ids = []
        
        for section in result.sections:
            for para in section.paragraphs:
                text = para.text if hasattr(para, 'text') else str(para.get('text', ''))
                all_text.append(text)
                section_ids.append(section.id)
        
        # Find references in combined text
        combined_text = ' '.join(all_text)
        
        # Figure references
        fig_patterns = [
            r'(?:Figure|Fig\.?|figure)\s*(\d+)',
            r'(?:see|shown|depicted)\s+(?:in\s+)?(?:Figure|Fig\.?)\s*(\d+)'
        ]
        
        for pattern in fig_patterns:
            for match in re.finditer(pattern, combined_text, re.IGNORECASE):
                fig_num = match.group(1)
                if fig_num not in reference_map['figures']:
                    reference_map['figures'][fig_num] = []
                # Add section context (simplified)
                reference_map['figures'][fig_num].append(f"referenced_in_text")
        
        # Similar for tables, code, equations...
        
        return reference_map
    
    def _extract_number_from_label(self, label: str) -> str:
        """Extract number from label string"""
        if not label:
            return ""
        match = re.search(r'\d+', str(label))
        return match.group() if match else ""
    
    async def _comprehensive_quality_validation(self, result: ExtractionResult) -> Dict[str, Any]:
        """Comprehensive quality validation"""
        metrics = {}
        
        # Content completeness
        metrics['extraction_coverage'] = self._calculate_coverage(result)
        
        # Content quality
        metrics['text_coherence'] = self._assess_text_coherence(result.sections)
        
        # Structure preservation
        metrics['structure_preservation'] = self._assess_structure_preservation(result)
        
        # Cross-reference accuracy
        metrics['reference_accuracy'] = self._assess_reference_accuracy(result)
        
        # Content type accuracy
        metrics['content_type_accuracy'] = await self._assess_content_type_accuracy(result)
        
        # Overall quality score
        metrics['overall_score'] = self._calculate_weighted_quality_score(metrics)
        
        return metrics
    
    def _assess_structure_preservation(self, result: ExtractionResult) -> float:
        """Assess how well document structure is preserved"""
        score = 0.0
        
        # Check section organization
        if result.sections:
            score += 0.3
            
            # Check if sections are in logical order
            page_order = [s.page_start for s in result.sections]
            if page_order == sorted(page_order):
                score += 0.2
        
        # Check metadata completeness
        if result.metadata and result.metadata.title != "Unknown":
            score += 0.2
        
        # Check reference structure
        if result.references:
            score += 0.2
        
        # Check content hierarchy
        if any(s.level for s in result.sections if hasattr(s, 'level')):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_reference_accuracy(self, result: ExtractionResult) -> float:
        """Assess accuracy of cross-references"""
        if not result.figures and not result.tables:
            return 1.0
        
        total_items = len(result.figures) + len(result.tables)
        referenced_items = 0
        
        for figure in result.figures:
            if getattr(figure, 'references', []):
                referenced_items += 1
        
        for table in result.tables:
            if getattr(table, 'references', []):
                referenced_items += 1
        
        return referenced_items / total_items if total_items > 0 else 1.0
    
    async def _assess_content_type_accuracy(self, result: ExtractionResult) -> float:
        """Assess accuracy of content type classification"""
        accuracy_scores = []
        
        # Assess table accuracy
        if result.tables:
            table_accuracy = 1.0 - await self._assess_table_false_positives(result.tables)
            accuracy_scores.append(table_accuracy)
        
        # Add assessments for other content types as needed
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _calculate_weighted_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'extraction_coverage': 0.25,
            'text_coherence': 0.25,
            'structure_preservation': 0.20,
            'reference_accuracy': 0.15,
            'content_type_accuracy': 0.15
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_final_status(self, quality_metrics: Dict[str, Any]) -> ExtractionStatus:
        """Determine final extraction status based on quality"""
        overall_score = quality_metrics.get('overall_score', 0.0)
        
        if overall_score >= self.QUALITY_THRESHOLDS['overall_score']:
            return ExtractionStatus.COMPLETED
        elif overall_score >= 0.5:
            return ExtractionStatus.PARTIAL
        else:
            return ExtractionStatus.FAILED