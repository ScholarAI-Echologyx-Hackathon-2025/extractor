# app/config.py
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings (Memory Optimized)"""
    
    # Application (hardcoded)
    app_name: str = "PDFExtractor"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # API (hardcoded)
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    api_prefix: str = "/api/v1"
    
    # External Services
    grobid_url: str = "http://localhost:8070"

    nougat_model_path: str = ""  # DISABLED: Heavy AI model removed for memory optimization
    
    # Enrichment APIs
    crossref_api_url: str = "https://api.crossref.org"
    crossref_email: Optional[str] = None
    openalex_api_url: str = "https://api.openalex.org"
    unpaywall_api_url: str = "https://api.unpaywall.org/v2"
    unpaywall_email: Optional[str] = None
    
    # Storage
    paper_folder: Path = Path("./paper")
    output_format: str = "json"
    keep_intermediate_files: bool = False  # Changed to False to save memory
    
    # Processing (hardcoded - only extraction_timeout is used)
    # max_workers: int = 2  # Not used in code
    extraction_timeout: int = 180  # Used in pipeline.py and main.py
    # ocr_language: str = "eng"  # Not used in code
    # use_gpu: bool = False  # Not used in code
    
    # OCR Configuration (Lightweight only)
    ocr_provider: str = "tesseract"  # Only Tesseract, no EasyOCR
    
    # Memory Optimization Settings
    enable_heavy_models: bool = False  # Disable AI models
    enable_table_transformer: bool = False  # Disable deep learning table extraction
    enable_nougat_ocr: bool = False  # Disable Nougat OCR
    enable_easyocr: bool = False  # Disable EasyOCR
    
    # Cache (not implemented - using in-memory storage)
    
    # RabbitMQ
    rabbitmq_user: Optional[str] = None
    rabbitmq_password: Optional[str] = None
    
    # PDF Storage (B2)
    b2_key_id: Optional[str] = None
    b2_application_key: Optional[str] = None
    b2_bucket_name: Optional[str] = None
    b2_bucket_id: Optional[str] = None
    
    # Gemini API not implemented
    
    # Cloudinary Configuration
    cloudinary_url: Optional[str] = None
    
    # Storage Configuration
    store_locally: bool = True
    upload_figures_to_cloud: bool = True       # upload at the end
    upload_concurrency: int = 6                # parallel Cloudinary uploads
    cloudinary_figures_folder: str = "scholarai/figures"
    
    # OCR Configuration
    ocr_space_api_key: Optional[str] = None
    use_ocr: bool = True
    
    # RabbitMQ Configuration
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: Optional[str] = None
    rabbitmq_password: Optional[str] = None
    rabbitmq_exchange: str = "scholarai.exchange"
    rabbitmq_extraction_queue: str = "scholarai.extraction.queue"
    rabbitmq_extraction_completed_queue: str = "scholarai.extraction.completed.queue"
    rabbitmq_extraction_routing_key: str = "scholarai.extraction"
    rabbitmq_extraction_completed_routing_key: str = "scholarai.extraction.completed"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create paper folder if it doesn't exist
        self.paper_folder.mkdir(exist_ok=True)


# Create global settings instance
settings = Settings()