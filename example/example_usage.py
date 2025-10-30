"""
example_usage.py - Example of using the PDF extraction system programmatically

This script demonstrates how to:
1. Extract content from a PDF
2. Access different types of extracted content
3. Save results in various formats
4. Use the extraction pipeline directly
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from app.config import settings
from services.pipeline import ExtractionPipeline
from models.schemas import ExtractionRequest
from utils.helpers import validate_pdf, get_pdf_info


async def example_basic_extraction():
    """Basic extraction example"""
    print("=" * 60)
    print("Example 1: Basic Extraction")
    print("=" * 60)
    
    # Find a PDF in the paper folder
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    if not pdf_files:
        print("No PDF found in paper folder")
        return
    
    pdf_path = pdf_files[0]
    print(f"Processing: {pdf_path.name}")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Create extraction request
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_text=True,
        extract_figures=True,
        extract_tables=True
    )
    
    # Run extraction
    result = await pipeline.extract(pdf_path, request)
    
    print(f"\nExtraction Status: {result.status}")
    print(f"Sections extracted: {len(result.sections)}")
    print(f"Figures extracted: {len(result.figures)}")
    print(f"Tables extracted: {len(result.tables)}")
    
    return result


async def example_text_extraction():
    """Example of working with extracted text"""
    print("\n" + "=" * 60)
    print("Example 2: Working with Text")
    print("=" * 60)
    
    # Get extraction result
    result = await example_basic_extraction()
    
    if not result:
        return
    
    # Access metadata
    print(f"\nTitle: {result.metadata.title}")
    if result.metadata.authors:
        authors = [a.name for a in result.metadata.authors]
        print(f"Authors: {', '.join(authors)}")
    
    # Access sections
    print("\nDocument Structure:")
    for section in result.sections[:5]:  # First 5 sections
        print(f"  - {section.title} (Pages {section.page_start}-{section.page_end})")
    
    # Extract all text
    all_text = []
    for section in result.sections:
        for para in section.paragraphs:
            if hasattr(para, 'text'):
                all_text.append(para.text)
            elif isinstance(para, dict):
                all_text.append(para.get('text', ''))
    
    full_text = '\n\n'.join(all_text)
    print(f"\nTotal text length: {len(full_text)} characters")
    
    # Save text to file
    text_file = settings.paper_folder / "extracted_text.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    print(f"Text saved to: {text_file}")


async def example_table_extraction():
    """Example of working with extracted tables"""
    print("\n" + "=" * 60)
    print("Example 3: Working with Tables")
    print("=" * 60)
    
    # Get extraction result
    result = await example_basic_extraction()
    
    if not result or not result.tables:
        print("No tables found")
        return
    
    print(f"\nFound {len(result.tables)} tables")
    
    # Process first table
    table = result.tables[0]
    print(f"\nTable: {table.label}")
    if table.caption:
        print(f"Caption: {table.caption}")
    
    # Convert to DataFrame
    if table.headers and table.rows:
        df = pd.DataFrame(table.rows, columns=table.headers[0])
        print("\nTable as DataFrame:")
        print(df.head())
        
        # Save as CSV
        csv_file = settings.paper_folder / "table_1.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nTable saved to: {csv_file}")


async def example_figure_extraction():
    """Example of working with extracted figures"""
    print("\n" + "=" * 60)
    print("Example 4: Working with Figures")
    print("=" * 60)
    
    # Get extraction result
    result = await example_basic_extraction()
    
    if not result or not result.figures:
        print("No figures found")
        return
    
    print(f"\nFound {len(result.figures)} figures")
    
    # List all figures with captions
    for i, figure in enumerate(result.figures[:5], 1):
        print(f"\n{i}. {figure.label}")
        if figure.caption:
            print(f"   Caption: {figure.caption[:100]}...")
        if figure.image_path:
            print(f"   Image: {figure.image_path}")
        print(f"   Page: {figure.page}")


async def example_code_extraction():
    """Example of working with extracted code blocks"""
    print("\n" + "=" * 60)
    print("Example 5: Working with Code Blocks")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Find PDF
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    if not pdf_files:
        return
    
    pdf_path = pdf_files[0]
    
    # Extract with code detection
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_code=True
    )
    
    result = await pipeline.extract(pdf_path, request)
    
    if not result.code_blocks:
        print("No code blocks found")
        return
    
    print(f"\nFound {len(result.code_blocks)} code blocks")
    
    # Group by language
    languages = {}
    for code in result.code_blocks:
        lang = code.language or 'unknown'
        if lang not in languages:
            languages[lang] = []
        languages[lang].append(code)
    
    print("\nCode blocks by language:")
    for lang, blocks in languages.items():
        print(f"  - {lang}: {len(blocks)} blocks")
    
    # Show first code block
    if result.code_blocks:
        first = result.code_blocks[0]
        print(f"\nFirst code block ({first.language}):")
        print("-" * 40)
        print(first.code[:200])
        print("-" * 40)


async def example_math_extraction():
    """Example of working with extracted equations"""
    print("\n" + "=" * 60)
    print("Example 6: Working with Mathematical Content")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Find PDF
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    if not pdf_files:
        return
    
    pdf_path = pdf_files[0]
    
    # Extract with equation detection
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_equations=True,
        use_ocr=True  # Important for scanned PDFs
    )
    
    result = await pipeline.extract(pdf_path, request)
    
    if not result.equations:
        print("No equations found")
        return
    
    print(f"\nFound {len(result.equations)} equations")
    
    # Separate inline and display equations
    inline = [eq for eq in result.equations if eq.inline]
    display = [eq for eq in result.equations if not eq.inline]
    
    print(f"  - Inline equations: {len(inline)}")
    print(f"  - Display equations: {len(display)}")
    
    # Show some equations
    print("\nSample equations:")
    for i, eq in enumerate(result.equations[:3], 1):
        eq_type = "inline" if eq.inline else "display"
        print(f"\n{i}. [{eq_type}] {eq.latex[:100]}")


async def example_advanced_extraction():
    """Example of advanced extraction with custom settings"""
    print("\n" + "=" * 60)
    print("Example 7: Advanced Extraction")
    print("=" * 60)
    
    # Find PDF
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    if not pdf_files:
        return
    
    pdf_path = pdf_files[0]
    
    # Get PDF info first
    pdf_info = get_pdf_info(pdf_path)
    
    # Determine if OCR is needed
    needs_ocr = not pdf_info['has_text'] or pdf_info['page_count'] > 50
    
    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {pdf_info['page_count']}")
    print(f"Has text: {pdf_info['has_text']}")
    print(f"Using OCR: {needs_ocr}")
    
    # Initialize pipeline
    pipeline = ExtractionPipeline()
    
    # Custom extraction request
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_text=True,
        extract_figures=True,
        extract_tables=True,
        extract_equations=True,
        extract_code=True,
        extract_references=True,
        use_ocr=needs_ocr,
        detect_entities=True,
        timeout=600  # 10 minutes for large PDFs
    )
    
    # Run extraction
    result = await pipeline.extract(pdf_path, request)
    
    # Generate comprehensive report
    report = {
        "pdf_info": pdf_info,
        "extraction_status": result.status,
        "extraction_coverage": result.extraction_coverage,
        "extraction_methods": result.extraction_methods,
        "confidence_scores": result.confidence_scores,
        "content_statistics": {
            "sections": len(result.sections),
            "figures": len(result.figures),
            "tables": len(result.tables),
            "equations": len(result.equations),
            "code_blocks": len(result.code_blocks),
            "references": len(result.references),
            "entities": len(result.entities)
        },
        "processing_time": result.processing_time
    }
    
    # Save report
    report_file = settings.paper_folder / "extraction_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\nExtraction Report:")
    print(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to: {report_file}")


async def main():
    """Run all examples"""
    print("PDF Extraction System - Usage Examples")
    print("=" * 60)
    
    # Check if PDF exists in paper folder
    pdf_files = list(settings.paper_folder.glob("*.pdf"))
    if not pdf_files:
        print("\n⚠️  No PDF found in paper folder!")
        print(f"Please place a PDF file in: {settings.paper_folder}")
        return
    
    print(f"\nFound PDF: {pdf_files[0].name}")
    print("\nRunning examples...")
    
    # Run examples
    try:
        # Basic extraction
        await example_basic_extraction()
        
        # Text extraction
        # await example_text_extraction()
        
        # Table extraction
        # await example_table_extraction()
        
        # Figure extraction
        # await example_figure_extraction()
        
        # Code extraction
        # await example_code_extraction()
        
        # Math extraction
        # await example_math_extraction()
        
        # Advanced extraction
        # await example_advanced_extraction()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())