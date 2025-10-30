# models/enums.py
from enum import Enum


class ExtractionStatus(str, Enum):
    """Extraction job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"


class EntityType(str, Enum):
    """Entity types found in papers"""
    DATASET = "dataset"
    CODE = "code"
    METHOD = "method"
    METRIC = "metric"
    MODEL = "model"
    TOOL = "tool"


class SectionType(str, Enum):
    """Common section types in academic papers"""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"