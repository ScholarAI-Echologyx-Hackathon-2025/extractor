#!/bin/bash

# PDF Extractor - Local Development Script
# Equivalent to the Makefile functionality

# Variables
PYTHON="python"
PIP="pip"
VENV="venv"
API_HOST="0.0.0.0"
API_PORT="8002"  # Updated to match the new default port

# Detect if running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    IS_WINDOWS=true
else
    IS_WINDOWS=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not activated. Activating..."
        source "$VENV/Scripts/activate"
    fi
}

# Help function
help() {
    echo "PDF Extractor - Available Commands:"
    echo ""
    echo "  ./scripts/local.sh setup        - Complete setup (venv, dependencies, services)"
    echo "  ./scripts/local.sh install      - Install Python dependencies"

    echo "  ./scripts/local.sh run          - Run the FastAPI server"
    echo "  ./scripts/local.sh test         - Run extraction test"
    echo "  ./scripts/local.sh extract      - Extract from paper folder"
    echo "  ./scripts/local.sh docker-up    - Start Docker services (GROBID, Redis, etc.)"
    echo "  ./scripts/local.sh docker-down  - Stop Docker services"
    echo "  ./scripts/local.sh clean        - Clean temporary files and cache"
    echo "  ./scripts/local.sh format       - Format code with black"
    echo "  ./scripts/local.sh lint         - Run linting checks"
    echo "  ./scripts/local.sh dev          - Development setup (install + docker-up + run)"
    echo "  ./scripts/local.sh quick-test   - Quick test with sample PDF"
    echo "  ./scripts/local.sh logs         - Monitor logs"
    echo "  ./scripts/local.sh create-dirs  - Create necessary directories"
    echo "  ./scripts/local.sh docs         - Show API documentation URLs"
    echo "  ./scripts/local.sh stats        - Show extraction statistics"
    echo "  ./scripts/local.sh check-deps   - Check system dependencies"
    echo ""
}

# Setup everything
setup() {
    print_info "Setting up PDF Extractor..."
    chmod +x setup.sh
    ./setup.sh
}

# Install dependencies
install() {
    print_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV" ]]; then
        print_info "Creating virtual environment..."
        $PYTHON -m venv "$VENV"
    fi
    
    # Activate virtual environment
    source "$VENV/Scripts/activate"
    
    # Upgrade pip properly
    print_info "Upgrading pip..."
    $PYTHON -m pip install --upgrade pip
    
    # Install PyTorch first (CPU version for Windows compatibility)
    print_info "Installing PyTorch..."
    $PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install all dependencies from requirements.txt
    print_info "Installing dependencies from requirements.txt..."
    $PIP install -r requirements.txt
    
    # Install spacy language model
    print_info "Installing spaCy language model..."
    $PYTHON -m spacy download en_core_web_sm
    
    print_success "Dependencies installed successfully!"
}

# Run the API server
run() {
    print_info "Starting FastAPI server..."
    check_venv
    uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --reload
}

# Run in production mode
run_prod() {
    print_info "Starting FastAPI server in production mode..."
    check_venv
    uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --workers 4
}

# Test extraction
test() {
    print_info "Running extraction test..."
    check_venv
    $PYTHON test_extraction.py
}

# Extract from paper folder
extract() {
    print_info "Extracting PDF from paper folder..."
    curl -X POST "http://localhost:$API_PORT/api/v1/extract-from-folder"
}

# Start Docker services
docker_up() {
    print_info "Starting Docker services..."
    if [[ "$IS_WINDOWS" == "true" ]]; then
        docker-compose.exe up -d
    else
        docker-compose up -d
    fi
    print_info "Waiting for services to be ready..."
    sleep 15
    
    # Check GROBID
    if curl -s http://localhost:8070/api/isalive > /dev/null; then
        print_success "GROBID is running"
    else
        print_error "GROBID failed to start"
    fi
}

# Stop Docker services
docker_down() {
    print_info "Stopping Docker services..."
    if [[ "$IS_WINDOWS" == "true" ]]; then
        docker-compose.exe down
    else
        docker-compose down
    fi
}

# View Docker logs
docker_logs() {
    if [[ "$IS_WINDOWS" == "true" ]]; then
        docker-compose.exe logs -f
    else
        docker-compose logs -f
    fi
}

# Clean temporary files
clean() {
    print_info "Cleaning temporary files..."
    
    # Remove Python cache files (Windows compatible)
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name ".DS_Store" -delete
    find . -type f -name "Thumbs.db" -delete
    
    # Remove test and build artifacts
    rm -rf .pytest_cache
    rm -rf .coverage
    rm -rf htmlcov
    rm -rf dist
    rm -rf build
    rm -rf *.egg-info
    
    print_success "Cleanup complete!"
}

# Format code
format() {
    print_info "Formatting code with black..."
    check_venv
    if [[ "$IS_WINDOWS" == "true" ]]; then
        python -m black . --exclude venv
    else
        black . --exclude venv
    fi
}

# Lint code
lint() {
    print_info "Running linting checks..."
    check_venv
    if [[ "$IS_WINDOWS" == "true" ]]; then
        python -m flake8 . --exclude venv,__pycache__ --max-line-length 120
        python -m mypy . --ignore-missing-imports
    else
        flake8 . --exclude venv,__pycache__ --max-line-length 120
        mypy . --ignore-missing-imports
    fi
}

# Run unit tests
test_unit() {
    print_info "Running unit tests..."
    check_venv
    if [[ "$IS_WINDOWS" == "true" ]]; then
        python -m pytest tests/ -v
    else
        pytest tests/ -v
    fi
}

# Check system dependencies
check_deps() {
    print_info "Checking system dependencies..."
    
    if command -v tesseract >/dev/null 2>&1; then
        print_success "✓ Tesseract installed"
    else
        print_error "✗ Tesseract not found"
    fi
    
    if command -v java >/dev/null 2>&1; then
        print_success "✓ Java installed"
    else
        print_error "✗ Java not found"
    fi
    
    if command -v docker >/dev/null 2>&1; then
        print_success "✓ Docker installed"
    else
        print_error "✗ Docker not found"
    fi
    
    if $PYTHON --version >/dev/null 2>&1; then
        print_success "✓ Python installed: $($PYTHON --version)"
    else
        print_error "✗ Python not found"
    fi
}

# Development setup
dev() {
    print_info "Setting up development environment..."
    install
    docker_up
    run
}

# Quick test with sample PDF
quick_test() {
    print_info "Quick test with sample PDF..."
    check_venv
    $PYTHON -c "
from utils.helpers import validate_pdf
from pathlib import Path
pdfs = list(Path('paper').glob('*.pdf'))
print(f'Found {len(pdfs)} PDFs')
for p in pdfs:
    print(f'  - {p.name}: Valid={validate_pdf(p)}')
"
}

# Monitor logs
logs() {
    if [[ -f "logs/pdf_extractor.log" ]]; then
        tail -f logs/pdf_extractor.log
    else
        print_warning "Log file not found. Creating logs directory..."
        mkdir -p logs
        touch logs/pdf_extractor.log
        tail -f logs/pdf_extractor.log
    fi
}

# Create directories
create_dirs() {
    print_info "Creating necessary directories..."
    mkdir -p paper paper/uploads paper/results paper/figures paper/tables paper/code paper/ocr_math logs
    print_success "Directories created!"
}

# Install development dependencies
install_dev() {
    print_info "Installing development dependencies..."
    check_venv
    $PIP install -r requirements-dev.txt
}



# Generate API documentation
docs() {
    print_info "API documentation available at:"
    echo "  http://localhost:$API_PORT/api/v1/docs (Swagger UI)"
    echo "  http://localhost:$API_PORT/api/v1/redoc (ReDoc)"
}

# Show extraction statistics for a PDF
stats() {
    print_info "Showing extraction statistics..."
    check_venv
    $PYTHON -c "
import json
from pathlib import Path
files = list(Path('paper').glob('*_extraction.json'))
if files:
    for f in files[:1]:
        data = json.load(open(f))
        stats = {k: len(v) if isinstance(v, list) else v for k, v in data.items() 
                if k in ['sections', 'figures', 'tables', 'equations', 'code_blocks', 'references']}
        print(f'\n{f.stem}:')
        print(json.dumps(stats, indent=2))
else:
    print('No extraction results found')
"
}

# Main script logic
main() {
    case "${1:-help}" in
        "help")
            help
            ;;
        "setup")
            setup
            ;;
        "install")
            install
            ;;

        "run")
            run
            ;;
        "run-prod")
            run_prod
            ;;
        "test")
            test
            ;;
        "extract")
            extract
            ;;
        "docker-up")
            docker_up
            ;;
        "docker-down")
            docker_down
            ;;
        "docker-logs")
            docker_logs
            ;;
        "clean")
            clean
            ;;
        "format")
            format
            ;;
        "lint")
            lint
            ;;
        "test-unit")
            test_unit
            ;;
        "check-deps")
            check_deps
            ;;
        "dev")
            dev
            ;;
        "quick-test")
            quick_test
            ;;
        "logs")
            logs
            ;;
        "create-dirs")
            create_dirs
            ;;
        "install-dev")
            install_dev
            ;;
        "docs")
            docs
            ;;
        "stats")
            stats
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            help
            exit 1
            ;;
    esac
}

# Run the main function with all arguments
main "$@"
