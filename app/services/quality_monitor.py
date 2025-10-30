# services/quality_monitor.py
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger
import pandas as pd
from collections import defaultdict, deque

from app.models.schemas import ExtractionResult, Table, Figure, Section
from app.config import settings


class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for extraction results"""
    overall_score: float
    extraction_coverage: float
    text_coherence: float
    table_accuracy: float
    figure_accuracy: float
    structure_preservation: float
    reference_consistency: float
    processing_efficiency: float
    error_rate: float
    
    # Detailed breakdowns
    table_false_positive_rate: float
    figure_detection_rate: float
    caption_accuracy: float
    text_flow_quality: float
    metadata_completeness: float
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def get_quality_level(self) -> QualityLevel:
        """Determine overall quality level"""
        if self.overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.overall_score >= 0.75:
            return QualityLevel.GOOD
        elif self.overall_score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif self.overall_score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED


@dataclass
class ExtractionStats:
    """Statistics for a single extraction"""
    pdf_path: str
    processing_time: float
    extraction_methods: List[str]
    quality_metrics: QualityMetrics
    timestamp: datetime
    errors: List[str]
    warnings: List[str]
    
    # Content counts
    sections_count: int
    tables_count: int
    figures_count: int
    equations_count: int
    references_count: int


class QualityMonitor:
    """
    Comprehensive quality monitoring system for PDF extraction
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.extraction_history = deque(maxlen=history_size)
        self.metrics_cache = {}
        self.performance_trends = defaultdict(list)
        
        # Quality benchmarks
        self.benchmarks = {
            'overall_score': {'excellent': 0.9, 'good': 0.75, 'acceptable': 0.6},
            'extraction_coverage': {'excellent': 0.85, 'good': 0.7, 'acceptable': 0.55},
            'text_coherence': {'excellent': 0.9, 'good': 0.75, 'acceptable': 0.6},
            'table_accuracy': {'excellent': 0.9, 'good': 0.8, 'acceptable': 0.65},
            'figure_accuracy': {'excellent': 0.85, 'good': 0.7, 'acceptable': 0.55},
            'table_false_positive_rate': {'excellent': 0.05, 'good': 0.15, 'acceptable': 0.3},
            'processing_efficiency': {'excellent': 0.9, 'good': 0.7, 'acceptable': 0.5}
        }
    
    async def evaluate_extraction(self, result: ExtractionResult) -> QualityMetrics:
        """Comprehensive evaluation of extraction result"""
        logger.info(f"Evaluating extraction quality for {result.pdf_path}")
        
        # Calculate individual metrics
        metrics = await self._calculate_comprehensive_metrics(result)
        
        # Add to history
        stats = ExtractionStats(
            pdf_path=result.pdf_path,
            processing_time=result.processing_time,
            extraction_methods=result.extraction_methods,
            quality_metrics=metrics,
            timestamp=datetime.utcnow(),
            errors=result.errors or [],
            warnings=result.warnings or [],
            sections_count=len(result.sections or []),
            tables_count=len(result.tables or []),
            figures_count=len(result.figures or []),
            equations_count=len(result.equations or []),
            references_count=len(result.references or [])
        )
        
        self.extraction_history.append(stats)
        self._update_performance_trends(stats)
        
        # Cache metrics
        self.metrics_cache[result.pdf_path] = metrics
        
        logger.info(f"Quality evaluation complete. Overall score: {metrics.overall_score:.2f}")
        return metrics
    
    async def _calculate_comprehensive_metrics(self, result: ExtractionResult) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        # 1. Content coverage and completeness
        coverage_score = self._calculate_coverage_score(result)
        
        # 2. Text quality and coherence
        text_coherence = self._evaluate_text_coherence(result.sections or [])
        
        # 3. Table extraction quality
        table_accuracy, table_fp_rate = self._evaluate_table_quality(result.tables or [])
        
        # 4. Figure extraction quality
        figure_accuracy, figure_detection_rate, caption_accuracy = self._evaluate_figure_quality(
            result.figures or []
        )
        
        # 5. Structure preservation
        structure_score = self._evaluate_structure_preservation(result)
        
        # 6. Reference consistency
        reference_consistency = self._evaluate_reference_consistency(result)
        
        # 7. Processing efficiency
        efficiency_score = self._calculate_processing_efficiency(result)
        
        # 8. Error analysis
        error_rate = self._calculate_error_rate(result)
        
        # 9. Additional metrics
        text_flow_quality = self._evaluate_text_flow_quality(result.sections or [])
        metadata_completeness = self._evaluate_metadata_completeness(result.metadata)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals({
            'coverage': coverage_score,
            'text_coherence': text_coherence,
            'table_accuracy': table_accuracy,
            'figure_accuracy': figure_accuracy,
            'structure': structure_score,
            'references': reference_consistency
        })
        
        # Overall weighted score
        overall_score = self._calculate_weighted_overall_score({
            'coverage': coverage_score,
            'text_coherence': text_coherence,
            'table_accuracy': table_accuracy,
            'figure_accuracy': figure_accuracy,
            'structure': structure_score,
            'references': reference_consistency,
            'efficiency': efficiency_score
        })
        
        return QualityMetrics(
            overall_score=overall_score,
            extraction_coverage=coverage_score,
            text_coherence=text_coherence,
            table_accuracy=table_accuracy,
            figure_accuracy=figure_accuracy,
            structure_preservation=structure_score,
            reference_consistency=reference_consistency,
            processing_efficiency=efficiency_score,
            error_rate=error_rate,
            table_false_positive_rate=table_fp_rate,
            figure_detection_rate=figure_detection_rate,
            caption_accuracy=caption_accuracy,
            text_flow_quality=text_flow_quality,
            metadata_completeness=metadata_completeness,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_coverage_score(self, result: ExtractionResult) -> float:
        """Calculate content coverage score"""
        score = 0.0
        
        # Metadata presence (15%)
        if result.metadata and result.metadata.title != "Unknown":
            score += 0.15
            if result.metadata.authors:
                score += 0.05
            if result.metadata.abstract:
                score += 0.05
        
        # Text content (40%)
        if result.sections:
            section_score = min(0.4, len(result.sections) * 0.05)
            score += section_score
            
            # Bonus for structured sections
            structured_sections = sum(1 for s in result.sections 
                                    if s.title and s.title != f"Section {s.id}")
            if structured_sections > 0:
                score += 0.05
        
        # Visual content (25%)
        if result.figures:
            figure_score = min(0.15, len(result.figures) * 0.03)
            score += figure_score
        
        if result.tables:
            table_score = min(0.1, len(result.tables) * 0.02)
            score += table_score
        
        # References (10%)
        if result.references:
            ref_score = min(0.1, len(result.references) * 0.01)
            score += ref_score
        
        # Equations and code (10%)
        if result.equations:
            eq_score = min(0.05, len(result.equations) * 0.01)
            score += eq_score
        
        if result.code_blocks:
            code_score = min(0.05, len(result.code_blocks) * 0.01)
            score += code_score
        
        return min(1.0, score)
    
    def _evaluate_text_coherence(self, sections: List[Section]) -> float:
        """Evaluate text coherence and readability"""
        if not sections:
            return 0.0
        
        coherence_scores = []
        
        for section in sections:
            if not section.paragraphs:
                continue
            
            # Extract text content
            section_text = ""
            for para in section.paragraphs:
                if isinstance(para, dict):
                    section_text += para.get('text', '') + " "
                else:
                    section_text += str(para.text) + " "
            
            if not section_text.strip():
                continue
            
            # Calculate coherence metrics
            coherence = self._calculate_text_coherence_metrics(section_text.strip())
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_text_coherence_metrics(self, text: str) -> float:
        """Calculate coherence metrics for text"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # 1. Sentence structure analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            # Average sentence length
            avg_sent_length = np.mean([len(s.split()) for s in sentences])
            if 8 <= avg_sent_length <= 30:  # Reasonable academic sentence length
                score += 0.25
            
            # Sentence length variance (good writing has variety)
            sent_lengths = [len(s.split()) for s in sentences]
            if len(sent_lengths) > 1:
                length_variance = np.var(sent_lengths)
                if 10 <= length_variance <= 100:  # Good variety without extremes
                    score += 0.15
        
        # 2. Vocabulary analysis
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.4:  # Good vocabulary diversity
                score += 0.2
        
        # 3. Punctuation and structure
        if '.' in text and ',' in text:  # Basic punctuation
            score += 0.1
        
        # 4. Capitalization patterns
        if any(c.isupper() for c in text) and any(c.islower() for c in text):
            score += 0.1
        
        # 5. Paragraph flow (check for connectors)
        connectors = ['however', 'therefore', 'moreover', 'furthermore', 'additionally',
                     'consequently', 'meanwhile', 'subsequently', 'nevertheless']
        if any(connector in text.lower() for connector in connectors):
            score += 0.1
        
        # 6. Academic language indicators
        academic_indicators = ['research', 'study', 'analysis', 'results', 'methodology',
                              'conclusion', 'evidence', 'significant', 'approach']
        if sum(1 for indicator in academic_indicators if indicator in text.lower()) >= 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_table_quality(self, tables: List[Table]) -> Tuple[float, float]:
        """Evaluate table extraction quality"""
        if not tables:
            return 1.0, 0.0  # Perfect score if no tables to evaluate
        
        valid_tables = 0
        false_positives = 0
        
        for table in tables:
            if self._is_valid_table(table):
                valid_tables += 1
            else:
                false_positives += 1
        
        accuracy = valid_tables / len(tables) if tables else 1.0
        fp_rate = false_positives / len(tables) if tables else 0.0
        
        return accuracy, fp_rate
    
    def _is_valid_table(self, table: Table) -> bool:
        """Check if table is genuinely tabular data"""
        if not table.rows or len(table.rows) < 2:
            return False
        
        # Check for tabular patterns
        all_cells = []
        if table.headers:
            for header_row in table.headers:
                all_cells.extend(str(cell) for cell in header_row)
        
        for row in table.rows:
            all_cells.extend(str(cell) for cell in row)
        
        if not all_cells:
            return False
        
        # Count numeric vs text cells
        numeric_cells = sum(1 for cell in all_cells 
                          if self._is_numeric_cell(str(cell)))
        
        # Tables typically have some numeric content
        numeric_ratio = numeric_cells / len(all_cells)
        
        # Check for prose-like content
        combined_text = ' '.join(all_cells)
        sentences = [s.strip() for s in combined_text.split('.') if s.strip()]
        
        # If too many complete sentences, likely prose
        if len(sentences) > len(table.rows):
            return False
        
        # Good tables have some numeric content and structured data
        return numeric_ratio > 0.1 or self._has_structured_data_patterns(all_cells)
    
    def _is_numeric_cell(self, cell: str) -> bool:
        """Check if cell contains numeric data"""
        cell = cell.strip()
        if not cell:
            return False
        
        # Direct number
        try:
            float(cell)
            return True
        except:
            pass
        
        # Percentage
        if cell.endswith('%'):
            try:
                float(cell[:-1])
                return True
            except:
                pass
        
        # Currency
        if any(cell.startswith(symbol) for symbol in ['$', '€', '£', '¥']):
            try:
                float(cell[1:].replace(',', ''))
                return True
            except:
                pass
        
        return False
    
    def _has_structured_data_patterns(self, cells: List[str]) -> bool:
        """Check for structured data patterns in cells"""
        # Date patterns
        date_pattern_count = sum(1 for cell in cells 
                               if self._looks_like_date(str(cell)))
        
        # Short consistent labels
        short_labels = [cell for cell in cells 
                       if len(str(cell).split()) <= 3 and str(cell).strip()]
        
        # Check for repeated structures
        cell_patterns = defaultdict(int)
        for cell in cells:
            cell_str = str(cell).strip()
            if cell_str:
                # Classify cell pattern
                if self._is_numeric_cell(cell_str):
                    cell_patterns['numeric'] += 1
                elif len(cell_str.split()) <= 2:
                    cell_patterns['short_text'] += 1
                else:
                    cell_patterns['long_text'] += 1
        
        # Good structured data has variety but not too much long text
        return (cell_patterns['long_text'] < len(cells) * 0.5 and
                (cell_patterns['numeric'] > 0 or date_pattern_count > 0))
    
    def _looks_like_date(self, cell: str) -> bool:
        """Check if cell looks like a date"""
        import re
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}'
        ]
        
        return any(re.search(pattern, cell) for pattern in date_patterns)
    
    def _evaluate_figure_quality(self, figures: List[Figure]) -> Tuple[float, float, float]:
        """Evaluate figure extraction quality"""
        if not figures:
            return 1.0, 1.0, 1.0  # Perfect scores if no figures
        
        valid_figures = 0
        figures_with_captions = 0
        
        for figure in figures:
            if self._is_valid_figure(figure):
                valid_figures += 1
            
            if figure.caption and len(figure.caption.strip()) > 5:
                figures_with_captions += 1
        
        accuracy = valid_figures / len(figures)
        detection_rate = min(1.0, len(figures) / 5)  # Assume ~5 figures per paper
        caption_accuracy = figures_with_captions / len(figures)
        
        return accuracy, detection_rate, caption_accuracy
    
    def _is_valid_figure(self, figure: Figure) -> bool:
        """Check if figure is valid"""
        # Check bounding box reasonableness
        if figure.bbox:
            width = figure.bbox.x2 - figure.bbox.x1
            height = figure.bbox.y2 - figure.bbox.y1
            area = width * height
            
            # Reasonable figure size
            if area < 1000 or area > 500000:
                return False
            
            # Reasonable aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                return False
        
        # Check if has image file
        if figure.image_path and Path(figure.image_path).exists():
            return True
        
        # Check label format
        if figure.label and any(keyword in figure.label.lower() 
                              for keyword in ['figure', 'fig', 'chart', 'graph']):
            return True
        
        return False
    
    def _evaluate_structure_preservation(self, result: ExtractionResult) -> float:
        """Evaluate how well document structure is preserved"""
        score = 0.0
        
        # Section organization (30%)
        if result.sections:
            score += 0.3
            
            # Logical page ordering
            page_numbers = [s.page_start for s in result.sections if s.page_start]
            if page_numbers and page_numbers == sorted(page_numbers):
                score += 0.1
            
            # Section hierarchy
            titled_sections = sum(1 for s in result.sections 
                                if s.title and s.title != f"Section {s.id}")
            if titled_sections > 0:
                score += 0.1
        
        # Metadata completeness (20%)
        if result.metadata:
            if result.metadata.title != "Unknown":
                score += 0.1
            if result.metadata.authors:
                score += 0.05
            if result.metadata.abstract:
                score += 0.05
        
        # Reference structure (20%)
        if result.references and len(result.references) > 5:
            score += 0.2
        
        # Content organization (30%)
        if result.figures or result.tables:
            score += 0.15
            
            # Check cross-references
            referenced_items = sum(1 for fig in (result.figures or []) 
                                 if hasattr(fig, 'references') and fig.references)
            referenced_items += sum(1 for table in (result.tables or []) 
                                  if hasattr(table, 'references') and table.references)
            
            total_items = len(result.figures or []) + len(result.tables or [])
            if total_items > 0:
                reference_ratio = referenced_items / total_items
                score += 0.15 * reference_ratio
        
        return min(1.0, score)
    
    def _evaluate_reference_consistency(self, result: ExtractionResult) -> float:
        """Evaluate consistency of cross-references"""
        if not result.sections:
            return 1.0
        
        # Find all references in text
        all_figure_refs = set()
        all_table_refs = set()
        
        for section in result.sections:
            for para in section.paragraphs:
                text = para.text if hasattr(para, 'text') else str(para.get('text', ''))
                
                import re
                fig_refs = re.findall(r'(?:Figure|Fig\.?)\s*(\d+)', text, re.IGNORECASE)
                table_refs = re.findall(r'Table\s*(\d+)', text, re.IGNORECASE)
                
                all_figure_refs.update(fig_refs)
                all_table_refs.update(table_refs)
        
        # Check consistency with actual figures/tables
        consistency_score = 0.0
        total_refs = len(all_figure_refs) + len(all_table_refs)
        
        if total_refs == 0:
            return 1.0  # No references to check
        
        # Check figure consistency
        figure_matches = 0
        for figure in (result.figures or []):
            if figure.label:
                fig_num = re.search(r'\d+', figure.label)
                if fig_num and fig_num.group() in all_figure_refs:
                    figure_matches += 1
        
        # Check table consistency  
        table_matches = 0
        for table in (result.tables or []):
            if table.label:
                table_num = re.search(r'\d+', table.label)
                if table_num and table_num.group() in all_table_refs:
                    table_matches += 1
        
        total_matches = figure_matches + table_matches
        consistency_score = total_matches / total_refs if total_refs > 0 else 1.0
        
        return consistency_score
    
    def _calculate_processing_efficiency(self, result: ExtractionResult) -> float:
        """Calculate processing efficiency score"""
        if not result.processing_time:
            return 0.5  # Neutral score if no timing info
        
        # Estimate content complexity
        content_complexity = (
            len(result.sections or []) * 0.5 +
            len(result.tables or []) * 2.0 +
            len(result.figures or []) * 1.5 +
            len(result.equations or []) * 1.0 +
            len(result.references or []) * 0.1
        )
        
        # Expected time based on complexity (rough heuristic)
        expected_time = max(10, content_complexity * 2)  # Minimum 10 seconds
        
        # Efficiency ratio
        if result.processing_time <= expected_time:
            return 1.0
        elif result.processing_time <= expected_time * 2:
            return 0.7
        elif result.processing_time <= expected_time * 3:
            return 0.4
        else:
            return 0.2
    
    def _calculate_error_rate(self, result: ExtractionResult) -> float:
        """Calculate error rate"""
        total_errors = len(result.errors or [])
        total_warnings = len(result.warnings or [])
        
        # Weight errors more than warnings
        error_score = total_errors * 1.0 + total_warnings * 0.5
        
        # Convert to rate (0 = no errors, 1 = many errors)
        if error_score == 0:
            return 0.0
        elif error_score <= 2:
            return 0.1
        elif error_score <= 5:
            return 0.3
        else:
            return 0.5
    
    def _evaluate_text_flow_quality(self, sections: List[Section]) -> float:
        """Evaluate quality of text flow reconstruction"""
        if not sections:
            return 1.0
        
        flow_score = 0.0
        
        # Check section ordering
        page_order_score = self._check_page_ordering(sections)
        flow_score += 0.4 * page_order_score
        
        # Check paragraph continuity
        continuity_score = self._check_paragraph_continuity(sections)
        flow_score += 0.3 * continuity_score
        
        # Check for fragmentation
        fragmentation_score = self._check_text_fragmentation(sections)
        flow_score += 0.3 * fragmentation_score
        
        return flow_score
    
    def _check_page_ordering(self, sections: List[Section]) -> float:
        """Check if sections are in correct page order"""
        page_starts = [s.page_start for s in sections if s.page_start]
        if not page_starts:
            return 1.0
        
        # Check if mostly in order
        ordered_count = sum(1 for i in range(len(page_starts) - 1) 
                          if page_starts[i] <= page_starts[i + 1])
        
        return ordered_count / max(1, len(page_starts) - 1)
    
    def _check_paragraph_continuity(self, sections: List[Section]) -> float:
        """Check paragraph continuity within sections"""
        continuity_scores = []
        
        for section in sections:
            if len(section.paragraphs) < 2:
                continue
            
            # Check for abrupt breaks
            paragraph_texts = []
            for para in section.paragraphs:
                text = para.text if hasattr(para, 'text') else str(para.get('text', ''))
                paragraph_texts.append(text.strip())
            
            # Simple continuity check: paragraphs shouldn't be too short
            avg_length = np.mean([len(text.split()) for text in paragraph_texts if text])
            
            if avg_length >= 10:  # Reasonable paragraph length
                continuity_scores.append(1.0)
            elif avg_length >= 5:
                continuity_scores.append(0.7)
            else:
                continuity_scores.append(0.3)
        
        return np.mean(continuity_scores) if continuity_scores else 1.0
    
    def _check_text_fragmentation(self, sections: List[Section]) -> float:
        """Check for excessive text fragmentation"""
        if not sections:
            return 1.0
        
        total_paragraphs = sum(len(s.paragraphs) for s in sections)
        total_sections = len(sections)
        
        # Reasonable paragraph-to-section ratio
        if total_sections == 0:
            return 1.0
        
        para_per_section = total_paragraphs / total_sections
        
        # Good ratio is 2-10 paragraphs per section
        if 2 <= para_per_section <= 10:
            return 1.0
        elif para_per_section > 10:
            return 0.8  # Slightly penalize very long sections
        else:
            return 0.5  # Penalize fragmented sections
    
    def _evaluate_metadata_completeness(self, metadata) -> float:
        """Evaluate completeness of extracted metadata"""
        if not metadata:
            return 0.0
        
        score = 0.0
        
        # Title (25%)
        if metadata.title and metadata.title != "Unknown":
            score += 0.25
        
        # Authors (25%)
        if metadata.authors and len(metadata.authors) > 0:
            score += 0.25
        
        # Abstract (20%)
        if metadata.abstract and len(metadata.abstract.strip()) > 50:
            score += 0.2
        
        # Publication info (15%)
        if metadata.year:
            score += 0.075
        if metadata.venue:
            score += 0.075
        
        # Identifiers (10%)
        if metadata.doi:
            score += 0.05
        if metadata.arxiv_id:
            score += 0.05
        
        # Keywords (5%)
        if metadata.keywords and len(metadata.keywords) > 0:
            score += 0.05
        
        return score
    
    def _calculate_confidence_intervals(self, metrics: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics"""
        intervals = {}
        
        for metric_name, value in metrics.items():
            # Simple confidence interval based on historical performance
            if len(self.extraction_history) > 10:
                # Get historical values for this metric
                historical_values = []
                for stats in self.extraction_history:
                    if hasattr(stats.quality_metrics, metric_name):
                        historical_values.append(getattr(stats.quality_metrics, metric_name))
                
                if historical_values:
                    std_dev = np.std(historical_values)
                    # 95% confidence interval (±1.96 * std_dev)
                    margin = 1.96 * std_dev
                    lower = max(0.0, value - margin)
                    upper = min(1.0, value + margin)
                    intervals[metric_name] = (lower, upper)
                else:
                    intervals[metric_name] = (max(0.0, value - 0.1), min(1.0, value + 0.1))
            else:
                # Default uncertainty for small sample size
                intervals[metric_name] = (max(0.0, value - 0.1), min(1.0, value + 0.1))
        
        return intervals
    
    def _calculate_weighted_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'coverage': 0.25,
            'text_coherence': 0.20,
            'table_accuracy': 0.15,
            'figure_accuracy': 0.15,
            'structure': 0.15,
            'references': 0.05,
            'efficiency': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _update_performance_trends(self, stats: ExtractionStats):
        """Update performance trend tracking"""
        metrics = stats.quality_metrics
        
        # Track key metrics over time
        trend_metrics = {
            'overall_score': metrics.overall_score,
            'processing_time': stats.processing_time,
            'table_accuracy': metrics.table_accuracy,
            'figure_accuracy': metrics.figure_accuracy,
            'error_rate': metrics.error_rate
        }
        
        for metric_name, value in trend_metrics.items():
            self.performance_trends[metric_name].append({
                'timestamp': stats.timestamp,
                'value': value,
                'pdf_path': stats.pdf_path
            })
            
            # Keep only recent data (last 100 extractions)
            if len(self.performance_trends[metric_name]) > 100:
                self.performance_trends[metric_name].pop(0)
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for recent period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_stats = [stats for stats in self.extraction_history 
                       if stats.timestamp >= cutoff_date]
        
        if not recent_stats:
            return {"message": "No recent extraction data available"}
        
        summary = {
            'period': f"Last {days} days",
            'total_extractions': len(recent_stats),
            'success_rate': self._calculate_success_rate(recent_stats),
            'average_metrics': self._calculate_average_metrics(recent_stats),
            'performance_trends': self._calculate_trend_analysis(recent_stats),
            'quality_distribution': self._calculate_quality_distribution(recent_stats),
            'common_issues': self._identify_common_issues(recent_stats),
            'recommendations': self._generate_recommendations(recent_stats)
        }
        
        return summary
    
    def _calculate_success_rate(self, stats_list: List[ExtractionStats]) -> float:
        """Calculate success rate"""
        successful = sum(1 for stats in stats_list 
                        if stats.quality_metrics.overall_score >= 0.6)
        return successful / len(stats_list) if stats_list else 0.0
    
    def _calculate_average_metrics(self, stats_list: List[ExtractionStats]) -> Dict[str, float]:
        """Calculate average metrics"""
        if not stats_list:
            return {}
        
        metrics = {
            'overall_score': np.mean([s.quality_metrics.overall_score for s in stats_list]),
            'extraction_coverage': np.mean([s.quality_metrics.extraction_coverage for s in stats_list]),
            'text_coherence': np.mean([s.quality_metrics.text_coherence for s in stats_list]),
            'table_accuracy': np.mean([s.quality_metrics.table_accuracy for s in stats_list]),
            'figure_accuracy': np.mean([s.quality_metrics.figure_accuracy for s in stats_list]),
            'processing_time': np.mean([s.processing_time for s in stats_list]),
            'error_rate': np.mean([s.quality_metrics.error_rate for s in stats_list])
        }
        
        return metrics
    
    def _calculate_trend_analysis(self, stats_list: List[ExtractionStats]) -> Dict[str, str]:
        """Calculate trend analysis"""
        if len(stats_list) < 5:
            return {"message": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        sorted_stats = sorted(stats_list, key=lambda x: x.timestamp)
        
        # Calculate trends for key metrics
        trends = {}
        
        # Overall score trend
        scores = [s.quality_metrics.overall_score for s in sorted_stats]
        trend = np.polyfit(range(len(scores)), scores, 1)[0]  # Linear trend
        
        if trend > 0.01:
            trends['overall_quality'] = "Improving"
        elif trend < -0.01:
            trends['overall_quality'] = "Declining"
        else:
            trends['overall_quality'] = "Stable"
        
        # Processing time trend
        times = [s.processing_time for s in sorted_stats]
        time_trend = np.polyfit(range(len(times)), times, 1)[0]
        
        if time_trend > 1.0:
            trends['processing_speed'] = "Slowing down"
        elif time_trend < -1.0:
            trends['processing_speed'] = "Speeding up"
        else:
            trends['processing_speed'] = "Stable"
        
        return trends
    
    def _calculate_quality_distribution(self, stats_list: List[ExtractionStats]) -> Dict[str, int]:
        """Calculate quality level distribution"""
        distribution = {level.value: 0 for level in QualityLevel}
        
        for stats in stats_list:
            quality_level = stats.quality_metrics.get_quality_level()
            distribution[quality_level.value] += 1
        
        return distribution
    
    def _identify_common_issues(self, stats_list: List[ExtractionStats]) -> List[str]:
        """Identify common issues across extractions"""
        issues = []
        
        # High table false positive rate
        high_fp_rate = sum(1 for s in stats_list 
                          if s.quality_metrics.table_false_positive_rate > 0.3)
        if high_fp_rate > len(stats_list) * 0.3:
            issues.append(f"High table false positive rate in {high_fp_rate} extractions")
        
        # Low text coherence
        low_coherence = sum(1 for s in stats_list 
                           if s.quality_metrics.text_coherence < 0.6)
        if low_coherence > len(stats_list) * 0.3:
            issues.append(f"Low text coherence in {low_coherence} extractions")
        
        # Processing timeout issues
        slow_processing = sum(1 for s in stats_list if s.processing_time > 300)  # 5 minutes
        if slow_processing > 0:
            issues.append(f"Slow processing (>5 min) in {slow_processing} extractions")
        
        # Frequent errors
        error_prone = sum(1 for s in stats_list if len(s.errors) > 2)
        if error_prone > len(stats_list) * 0.2:
            issues.append(f"Multiple errors in {error_prone} extractions")
        
        return issues
    
    def _generate_recommendations(self, stats_list: List[ExtractionStats]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        avg_metrics = self._calculate_average_metrics(stats_list)
        
        # Table accuracy recommendations
        if avg_metrics.get('table_accuracy', 1.0) < 0.7:
            recommendations.append(
                "Consider adjusting table detection thresholds or improving validation pipeline"
            )
        
        # Text coherence recommendations
        if avg_metrics.get('text_coherence', 1.0) < 0.7:
            recommendations.append(
                "Implement better text flow reconstruction algorithms"
            )
        
        # Processing time recommendations
        if avg_metrics.get('processing_time', 0) > 240:  # 4 minutes
            recommendations.append(
                "Optimize extraction pipeline for better performance"
            )
        
        # Coverage recommendations
        if avg_metrics.get('extraction_coverage', 1.0) < 0.7:
            recommendations.append(
                "Improve content detection methods to increase coverage"
            )
        
        return recommendations
    
    async def save_quality_report(self, output_path: Path):
        """Save comprehensive quality report"""
        report = {
            'generation_time': datetime.utcnow().isoformat(),
            'total_extractions': len(self.extraction_history),
            'recent_performance': self.get_performance_summary(days=7),
            'all_time_performance': self.get_performance_summary(days=365) if self.extraction_history else {},
            'benchmarks': self.benchmarks,
            'detailed_history': [
                {
                    'pdf_path': stats.pdf_path,
                    'timestamp': stats.timestamp.isoformat(),
                    'quality_metrics': asdict(stats.quality_metrics),
                    'processing_time': stats.processing_time,
                    'content_counts': {
                        'sections': stats.sections_count,
                        'tables': stats.tables_count,
                        'figures': stats.figures_count,
                        'equations': stats.equations_count,
                        'references': stats.references_count
                    },
                    'errors': stats.errors,
                    'warnings': stats.warnings
                }
                for stats in list(self.extraction_history)[-50:]  # Last 50 extractions
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Quality report saved to {output_path}")


# Global quality monitor instance
quality_monitor = QualityMonitor()
