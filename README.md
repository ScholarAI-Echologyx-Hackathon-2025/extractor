# PDF Extractor - Advanced Academic Paper Extraction System

A robust, multi-method PDF extraction system designed to achieve near 100% extraction coverage from academic papers. The system uses multiple extraction techniques with fallback strategies to ensure maximum content recovery.

## Features

- **Multi-Method Extraction**: Combines GROBID, Table Transformer, Nougat, Tesseract OCR, and computer vision techniques
- **Comprehensive Content Types**: Extracts text, figures, tables, equations, code blocks, references, and named entities
- **High Accuracy**: Multiple extraction methods with intelligent fallback strategies
- **OCR Support**: Automatic detection and processing of scanned PDFs
- **Cloud Storage Integration**: Direct extraction from Backblaze B2 storage URLs
- **Modular Architecture**: Clean, extensible codebase with separate extractors
- **REST API**: FastAPI-based API with async support
- **Quality Metrics**: Extraction coverage and confidence scoring

## Extraction Capabilities

### 1. Text & Structure (GROBID)
- Title, authors, abstract
- Hierarchical sections
- Paragraphs with page locations
- References with metadata

### 2. Figures (PyMuPDF + CV)
- Figure detection and extraction
- Caption extraction
- Multi-method deduplication
- Support for embedded and vector graphics

### 3. Tables (Table Transformer + PDFPlumber)
- Complex table structure recognition
- Header and cell extraction
- CSV and HTML output
- Deep learning-based detection

### 4. Mathematical Content (Nougat + Pattern Matching)
- LaTeX equation extraction
- Inline and display equations
- Mathematical OCR for scanned content

### 5. Code Blocks (Pattern Matching + Visual Detection)
- Language detection
- Algorithm/pseudocode extraction
- Syntax-aware extraction

### 6. OCR (Nougat + Tesseract)
- Automatic scanned PDF detection
- Mathematical formula OCR
- Layout-aware text extraction

### 7. Cloud Storage Integration (Backblaze B2)
- Direct extraction from B2 download URLs
- Authenticated downloads with fallback support
- Automatic PDF validation and cleanup

## Installation

### Prerequisites

- Python 3.10+

- Docker (for GROBID)
- Tesseract OCR

### Quick Setup

1. Clone the repository:
```bash
git clone <repository>
cd pdf-extractor
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create virtual environment
- Install Python dependencies
- Download required models
- Set up directories
- Start GROBID service
- Create configuration files

### Manual Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Start GROBID service:
```bash
docker-compose up -d grobid
```

4. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng default-jre

# macOS
brew install tesseract
brew install openjdk
```

## Usage

### Method 1: Paper Folder Extraction (As Specified)

1. Place your PDF in the `paper` folder:
```bash
cp your_paper.pdf paper/
```

2. Run extraction:
```bash
# Start the API server
uvicorn app.main:app --reload

# In another terminal, trigger extraction
curl -X POST http://localhost:8000/api/v1/extract-from-folder
```

3. Results will be saved in the `paper` folder:
- `paper/<filename>_extraction.json` - Full extraction results
- `paper/<filename>_summary.txt` - Human-readable summary

### Method 2: API Upload

1. Start the API server:
```bash
uvicorn app.main:app --reload
```

2. Upload and extract:
```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@your_paper.pdf" \
  -F "extract_text=true" \
```

### Method 3: B2 Cloud Storage Extraction

Extract PDFs directly from Backblaze B2 URLs:

1. Configure B2 credentials in `.env`:
```env
B2_KEY_ID=your_b2_key_id
B2_APPLICATION_KEY=your_b2_application_key
B2_BUCKET_NAME=your_bucket_name
```

2. Extract from B2 URL:
```bash
curl -X POST http://localhost:8002/api/v1/extract-from-b2 \
  -H "Content-Type: application/json" \
  -d '{
    "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=your_file_id",
    "extract_text": true,
    "extract_figures": true,
    "async_processing": false
  }'
```

3. For async processing:
```python
import requests

# Start extraction
response = requests.post("http://localhost:8002/api/v1/extract-from-b2", json={
    "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=your_file_id",
    "async_processing": True
})

job_id = response.json()["job_id"]

# Check status
status_response = requests.get(f"http://localhost:8002/api/v1/status/{job_id}")
print(status_response.json())
```

See `docs/b2_extraction_api.md` for complete API documentation and `example/b2_extraction_example.py` for usage examples.
```

3. Check extraction status:
```bash
curl http://localhost:8000/api/v1/status/{job_id}
```

4. Get results:
```bash
curl http://localhost:8000/api/v1/result/{job_id}
```

### API Documentation

Access the interactive API documentation at:
- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

## Project Structure

```
pdf-extractor/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   └── config.py          # Configuration settings
├── models/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic models
│   └── enums.py          # Enumerations
├── services/
│   ├── __init__.py
│   ├── pipeline.py        # Main extraction pipeline
│   └── extractors/
│       ├── __init__.py
│       ├── grobid_extractor.py      # GROBID integration
│       ├── figure_extractor.py      # Figure extraction
│       ├── table_extractor.py       # Table extraction
│       ├── ocr_math_extractor.py    # OCR and math
│       └── code_extractor.py        # Code extraction
├── utils/
│   ├── __init__.py
│   ├── exceptions.py      # Custom exceptions
│   └── helpers.py         # Utility functions
├── paper/                 # PDF storage folder
├── logs/                  # Application logs
├── docker-compose.yml     # Docker services
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
└── README.md             # This file
```

## Configuration

Edit `.env` file to configure:

```env
# External Services
GROBID_URL=http://localhost:8070

NOUGAT_MODEL_PATH=facebook/nougat-base

# Processing
MAX_WORKERS=4
EXTRACTION_TIMEOUT=300
OCR_LANGUAGE=eng
USE_GPU=False

# Storage
PAPER_FOLDER=./paper
KEEP_INTERMEDIATE_FILES=True
```

## Extraction Methods Priority

The system uses these extraction methods in order of preference:

1. **GROBID**: Primary method for text and structure

3. **Table Transformer**: Primary method for complex tables
4. **Nougat**: Primary method for mathematical OCR
5. **PyMuPDF**: Fallback for embedded content
6. **Tesseract**: General OCR fallback
7. **Computer Vision**: Last resort for visual detection

## Quality Metrics

The system provides quality metrics for each extraction:

- **Coverage Score**: Percentage of expected content extracted (0-100%)
- **Confidence Scores**: Per-component confidence levels
- **Extraction Status**: COMPLETED, PARTIAL, or FAILED
- **Processing Time**: Time taken for extraction

## Troubleshooting

### GROBID not starting
```bash
# Check Docker logs
docker logs pdf_extractor_grobid

# Restart service
docker-compose restart grobid
```

### OCR not working
```bash
# Check Tesseract installation
tesseract --version

# Install language packs
sudo apt-get install tesseract-ocr-eng
```

### Low extraction quality
- Ensure PDF is not corrupted
- Check if PDF is scanned (enable OCR)
- Increase timeout for large PDFs
- Check available system memory

## Advanced Usage

### Custom Extraction Pipeline

```python
from services.pipeline import ExtractionPipeline
from models.schemas import ExtractionRequest

pipeline = ExtractionPipeline()
request = ExtractionRequest(
    pdf_path="paper/your_paper.pdf",
    extract_text=True,
    extract_figures=True,
    # ... configure as needed
)

result = await pipeline.extract(pdf_path, request)
```

### Adding New Extractors

1. Create new extractor in `services/extractors/`
2. Implement extraction logic
3. Register in pipeline
4. Update schemas if needed

## Performance

Typical extraction times on standard hardware:
- 10-page paper: 15-30 seconds
- 30-page paper: 45-90 seconds
- 100-page thesis: 3-5 minutes

Factors affecting performance:
- PDF complexity
- OCR requirements
- Number of figures/tables
- System resources

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Tests are included
- Documentation is updated
- Extraction coverage is maintained or improved

## License

[Your License Here]

## Acknowledgments

- GROBID for scientific document parsing

- Microsoft for Table Transformer
- Meta for Nougat
- All open-source contributors