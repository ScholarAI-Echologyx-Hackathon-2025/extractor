# models/schemas.py
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID, uuid4
from .enums import ExtractionStatus, EntityType, SectionType


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int
    confidence: Optional[float] = None


class Paragraph(BaseModel):
    """Text paragraph with location"""
    text: str
    page: int
    bbox: Optional[BoundingBox] = None
    style: Optional[Dict[str, Any]] = None  # font, size, etc.


class Section(BaseModel):
    """Document section"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: Optional[str] = None  # e.g., "1.1"
    title: str
    type: SectionType = SectionType.OTHER
    level: int = 1  # heading level
    page_start: int
    page_end: int
    paragraphs: List[Paragraph] = []
    subsections: List['Section'] = []


class Figure(BaseModel):
    """Extracted figure"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: Optional[str] = None  # e.g., "Figure 1"
    caption: Optional[str] = None
    page: int
    bbox: BoundingBox
    image_path: Optional[str] = None  # path to extracted image
    thumbnail_path: Optional[str] = None
    type: str = "figure"  # figure, chart, diagram, etc.
    references: List[str] = []  # sections that reference this figure
    
    # OCR extracted text for LLM processing
    ocr_text: Optional[str] = None  # text extracted from the figure image
    ocr_confidence: Optional[float] = None  # OCR confidence score (0-1)
    
    # Enhanced extraction metadata
    extraction_method: Optional[str] = None  # pymupdf, cv_contour, etc.
    validation_scores: Optional[Dict[str, float]] = None  # validation metrics
    confidence: Optional[float] = None  # overall confidence score


class Table(BaseModel):
    """Extracted table"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: Optional[str] = None  # e.g., "Table 1"
    caption: Optional[str] = None
    page: int
    bbox: BoundingBox
    headers: List[List[str]] = []
    rows: List[List[str]] = []
    csv_path: Optional[str] = None
    html: Optional[str] = None
    structure: Optional[Dict[str, Any]] = None  # detailed structure
    references: List[str] = []
    
    # Enhanced extraction metadata
    extraction_method: Optional[str] = None  # pdfplumber, transformer, pymupdf
    validation_scores: Optional[Dict[str, float]] = None  # validation metrics
    image_path: Optional[str] = None  # path to table image if extracted


class CodeBlock(BaseModel):
    """Extracted code block"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    language: Optional[str] = None
    code: str
    page: int
    bbox: Optional[BoundingBox] = None
    context: Optional[str] = None  # surrounding text
    line_numbers: bool = False


class Equation(BaseModel):
    """Mathematical equation"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    label: Optional[str] = None
    latex: str
    mathml: Optional[str] = None
    page: int
    bbox: Optional[BoundingBox] = None
    inline: bool = False


class Reference(BaseModel):
    """Bibliographic reference"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_text: str
    title: Optional[str] = None
    authors: List[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[HttpUrl] = None
    arxiv_id: Optional[str] = None
    
    # Enrichment data
    crossref_data: Optional[Dict[str, Any]] = None
    openalex_data: Optional[Dict[str, Any]] = None
    unpaywall_data: Optional[Dict[str, Any]] = None
    
    # Citation context
    cited_by_sections: List[str] = []
    citation_count: int = 0


class Author(BaseModel):
    """Paper author"""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class Entity(BaseModel):
    """Named entity in paper"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EntityType
    name: str
    uri: Optional[HttpUrl] = None
    page: int
    context: str
    confidence: float = 1.0


class Metadata(BaseModel):
    """Paper metadata"""
    title: str
    authors: List[Author] = []
    abstract: Optional[str] = None
    keywords: List[str] = []
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    license: Optional[str] = None
    language: str = "en"
    page_count: int = 0


class ExtractionResult(BaseModel):
    """Complete extraction result"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    pdf_path: str
    pdf_hash: str
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core content
    metadata: Metadata
    sections: List[Section] = []
    figures: List[Figure] = []
    tables: List[Table] = []
    equations: List[Equation] = []
    code_blocks: List[CodeBlock] = []
    references: List[Reference] = []
    entities: List[Entity] = []
    
    # Processing metadata
    status: ExtractionStatus = ExtractionStatus.PENDING
    extraction_methods: List[str] = []
    processing_time: Optional[float] = None
    errors: List[str] = []
    warnings: List[str] = []
    
    # Quality metrics
    extraction_coverage: Optional[float] = None  # 0-100%
    confidence_scores: Dict[str, float] = {}
    quality_metrics: Optional[Dict[str, Any]] = None  # Enhanced quality assessment metrics
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class ExtractionRequest(BaseModel):
    """API request for extraction"""
    pdf_path: Optional[str] = None
    pdf_url: Optional[HttpUrl] = None
    pdf_base64: Optional[str] = None
    
    # Extraction options
    extract_text: bool = True
    extract_figures: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    extract_code: bool = True
    extract_references: bool = True
    
    # Enhancement options
    use_ocr: bool = True
    enrich_references: bool = True
    detect_entities: bool = True
    
    # Processing options
    force_reprocess: bool = False
    keep_intermediate: bool = True
    timeout: Optional[int] = None


class ExtractionResponse(BaseModel):
    """API response for extraction"""
    job_id: str
    status: ExtractionStatus
    message: Optional[str] = None
    result: Optional[ExtractionResult] = None
    progress: Optional[float] = None  # 0-100%
    estimated_time_remaining: Optional[int] = None  # seconds


class B2ExtractionRequest(BaseModel):
    """Request for extracting PDF from B2 URL"""
    b2_url: str = Field(..., description="Backblaze B2 download URL with fileId parameter")
    
    # Extraction options (same as ExtractionRequest)
    extract_text: bool = True
    extract_figures: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    extract_code: bool = True
    extract_references: bool = True
    use_ocr: bool = True
    detect_entities: bool = True
    
    # Processing options
    async_processing: bool = False
    timeout: Optional[int] = None


class B2ExtractionResponse(BaseModel):
    """Response for B2 extraction request"""
    job_id: str
    status: ExtractionStatus
    message: Optional[str] = None
    b2_url: str
    result: Optional[ExtractionResult] = None
    progress: Optional[float] = None  # 0-100%
    estimated_time_remaining: Optional[int] = None  # seconds


# Update Section model to handle nested references
Section.model_rebuild()