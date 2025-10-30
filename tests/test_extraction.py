"""
test_extraction.py - Test script for enhanced PDF extraction workflow

Usage:
    python test_extraction.py [pdf_path]
    
If no PDF path is provided, it will look for a PDF in the paper folder.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.extraction_handler import enhanced_extraction_handler
from app.services.config_manager import config_manager, DocumentType, ProcessingProfile
from app.services.quality_monitor import quality_monitor
from app.models.schemas import ExtractionRequest
from app.utils.helpers import validate_pdf, get_pdf_info, create_extraction_summary
from loguru import logger


async def test_enhanced_extraction(pdf_path: Optional[Path] = None):
    """
    Test enhanced PDF extraction with full workflow including:
    - Configuration management
    - Quality monitoring
    - Adaptive processing
    - Performance tracking
    """
    print("=" * 80)
    print("Enhanced PDF Extraction Test - Full Workflow")
    print("=" * 80)
    
    # Find PDF to process
    if pdf_path and pdf_path.exists():
        target_pdf = pdf_path
    else:
        # Look in paper folder
        pdf_files = list(settings.paper_folder.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF found in paper folder")
            print(f"   Please place a PDF in: {settings.paper_folder}")
            return
        
        if len(pdf_files) > 1:
            print("⚠️  Multiple PDFs found:")
            for i, pdf in enumerate(pdf_files):
                print(f"   {i+1}. {pdf.name}")
            choice = input("Select PDF number (default=1): ").strip()
            idx = int(choice) - 1 if choice.isdigit() else 0
            target_pdf = pdf_files[idx]
        else:
            target_pdf = pdf_files[0]
    
    print(f"\n📄 Processing: {target_pdf.name}")
    
    # Validate PDF
    print("\n🔍 Validating PDF...")
    if not validate_pdf(target_pdf):
        print("❌ Invalid PDF file")
        return
    
    # Get PDF info
    pdf_info = get_pdf_info(target_pdf)
    print(f"   Pages: {pdf_info['page_count']}")
    print(f"   Size: {pdf_info['file_size'] / 1024 / 1024:.2f} MB")
    print(f"   Has Text: {pdf_info['has_text']}")
    print(f"   Has Images: {pdf_info['has_images']}")
    print(f"   Encrypted: {pdf_info['encrypted']}")
    
    if pdf_info['metadata'].get('title'):
        print(f"   Title: {pdf_info['metadata']['title']}")
    
    # Initialize enhanced workflow components
    print("\n🚀 Initializing enhanced extraction workflow...")
    
    # Get optimal configuration based on document characteristics
    print("\n⚙️  Determining optimal configuration...")
    document_type = DocumentType.ACADEMIC_PAPER  # Default assumption
    processing_profile = ProcessingProfile.BALANCED
    
    # Analyze document characteristics
    if pdf_info['page_count'] > 20:
        document_type = DocumentType.THESIS
    elif pdf_info['has_images'] and pdf_info['page_count'] > 10:
        document_type = DocumentType.TECHNICAL_REPORT
    
    config = await config_manager.get_optimal_config(
        document_type=document_type,
        processing_profile=processing_profile,
        quality_target=0.8,
        time_constraint=600  # 10 minutes
    )
    
    print(f"   Document Type: {document_type.value}")
    print(f"   Processing Profile: {processing_profile.value}")
    print(f"   Target Quality: {config.target_quality:.2f}")
    print(f"   Max Processing Time: {config.max_processing_time}s")
    print(f"   Table Validation Threshold: {config.table_validation_threshold:.2f}")
    print(f"   Figure Validation Threshold: {config.figure_validation_threshold:.2f}")
    
    # Create extraction request with enhanced parameters
    request = ExtractionRequest(
        pdf_path=str(target_pdf),
        extract_text=True,
        extract_figures=True,
        extract_tables=True,
        extract_equations=True,
        extract_code=True,
        extract_references=True,
        use_ocr=config.ocr_enabled,
        detect_entities=True,
        timeout=config.max_processing_time
    )
    
    # Run enhanced extraction
    print("\n⚙️  Starting enhanced extraction...")
    print("   This may take a few minutes depending on PDF size and complexity...")
    
    start_time = datetime.now()
    
    try:
        # Create a mock message for the enhanced extraction handler
        mock_message = {
            'jobId': f'test_{target_pdf.stem}',
            'paperId': target_pdf.stem,
            'correlationId': 'test_correlation',
            'b2Url': f'file://{target_pdf.absolute()}',
            'qualityTarget': config.target_quality,
            'extractText': request.extract_text,
            'extractFigures': request.extract_figures,
            'extractTables': request.extract_tables,
            'extractEquations': request.extract_equations,
            'extractCode': request.extract_code,
            'extractReferences': request.extract_references,
            'useOcr': request.use_ocr,
            'detectEntities': request.detect_entities,
            'processingHints': {
                'document_type': document_type.value,
                'processing_profile': processing_profile.value,
                'config': config.__dict__
            }
        }
        
        # Use the pipeline directly for testing (since we don't have B2 service for local files)
        from app.services.pipeline import ExtractionPipeline
        pipeline = ExtractionPipeline()
        result = await pipeline.extract(target_pdf, request)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Enhanced extraction completed in {elapsed_time:.2f} seconds")
        
        # Quality evaluation
        print("\n🔍 Evaluating extraction quality...")
        quality_metrics = await quality_monitor.evaluate_extraction(result)
        
        print(f"\n📊 Enhanced Extraction Results:")
        print(f"   Status: {result.status}")
        print(f"   Overall Quality Score: {quality_metrics.overall_score:.3f}")
        print(f"   Quality Level: {quality_metrics.get_quality_level().value}")
        print(f"   Coverage: {result.extraction_coverage}%")
        print(f"   Methods Used: {', '.join(result.extraction_methods)}")
        
        print("\n📝 Content Extracted:")
        print(f"   Sections: {len(result.sections)}")
        print(f"   Figures: {len(result.figures)}")
        print(f"   Tables: {len(result.tables)}")
        print(f"   Equations: {len(result.equations)}")
        print(f"   Code Blocks: {len(result.code_blocks)}")
        print(f"   References: {len(result.references)}")
        print(f"   Entities: {len(result.entities)}")
        
        # Detailed quality metrics
        print("\n🎯 Quality Metrics:")
        print(f"   Text Coherence: {quality_metrics.text_coherence:.3f}")
        print(f"   Table Accuracy: {quality_metrics.table_accuracy:.3f}")
        print(f"   Figure Accuracy: {quality_metrics.figure_accuracy:.3f}")
        print(f"   Structure Preservation: {quality_metrics.structure_preservation:.3f}")
        print(f"   Reference Consistency: {quality_metrics.reference_consistency:.3f}")
        print(f"   Processing Efficiency: {quality_metrics.processing_efficiency:.3f}")
        print(f"   Error Rate: {quality_metrics.error_rate:.3f}")
        
        # Show enhanced metadata for extracted content
        if result.figures:
            print(f"\n🖼️  Figure Details:")
            for i, fig in enumerate(result.figures[:3]):  # Show first 3
                print(f"   {i+1}. {fig.label or f'Figure {i+1}'}")
                print(f"      Method: {fig.extraction_method or 'unknown'}")
                confidence_str = f"{fig.confidence:.3f}" if fig.confidence is not None else "N/A"
                print(f"      Confidence: {confidence_str}")
                if fig.validation_scores:
                    print(f"      Validation: {fig.validation_scores}")
        
        if result.tables:
            print(f"\n📊 Table Details:")
            for i, table in enumerate(result.tables[:3]):  # Show first 3
                print(f"   {i+1}. {table.label or f'Table {i+1}'}")
                print(f"      Method: {table.extraction_method or 'unknown'}")
                if table.validation_scores:
                    print(f"      Validation: {table.validation_scores}")
        
        # Show quality metrics if available
        if result.quality_metrics:
            print(f"\n📈 Quality Assessment:")
            for key, value in result.quality_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        
        # Show any errors or warnings
        if result.errors:
            print("\n⚠️  Errors:")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        # Save results with enhanced metadata
        output_path = settings.paper_folder / f"{target_pdf.stem}_enhanced_extraction.json"
        result_dict = result.dict()
        result_dict['quality_metrics'] = quality_metrics.__dict__
        result_dict['configuration_used'] = config.__dict__
        result_dict['document_analysis'] = {
            'document_type': document_type.value,
            'processing_profile': processing_profile.value,
            'pdf_info': pdf_info
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Enhanced results saved to: {output_path}")
        
        # Create and save summary
        summary = create_extraction_summary(result_dict)
        summary_path = settings.paper_folder / f"{target_pdf.stem}_enhanced_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📄 Enhanced summary saved to: {summary_path}")
        
        # Show sample extracted content
        print("\n📖 Sample Extracted Content:")
        
        # Show title and abstract
        if result.metadata.title != "Unknown":
            print(f"\nTitle: {result.metadata.title}")
        
        if result.metadata.abstract:
            print(f"\nAbstract: {result.metadata.abstract[:200]}...")
        
        # Show first section
        if result.sections:
            first_section = result.sections[0]
            print(f"\nFirst Section: {first_section.title}")
            if first_section.paragraphs:
                first_para = first_section.paragraphs[0]
                text = first_para.text if hasattr(first_para, 'text') else first_para.get('text', '')
                print(f"   {text[:200]}...")
        
        # Show first figure with enhanced metadata
        if result.figures:
            first_figure = result.figures[0]
            print(f"\nFirst Figure: {first_figure.label}")
            print(f"   Extraction Method: {first_figure.extraction_method or 'unknown'}")
            confidence_str = f"{first_figure.confidence:.3f}" if first_figure.confidence is not None else "N/A"
            print(f"   Confidence: {confidence_str}")
            if first_figure.caption:
                print(f"   Caption: {first_figure.caption[:100]}...")
        
        # Show first table with enhanced metadata
        if result.tables:
            first_table = result.tables[0]
            print(f"\nFirst Table: {first_table.label}")
            print(f"   Extraction Method: {first_table.extraction_method or 'unknown'}")
            if first_table.caption:
                print(f"   Caption: {first_table.caption[:100]}...")
            if first_table.headers:
                print(f"   Headers: {first_table.headers[0][:5]}...")
        
        # Show first equation
        if result.equations:
            first_eq = result.equations[0]
            print(f"\nFirst Equation: {first_eq.latex[:50]}...")
        
        # Show first code block
        if result.code_blocks:
            first_code = result.code_blocks[0]
            print(f"\nFirst Code Block ({first_code.language or 'unknown'}):")
            print(f"   {first_code.code[:100]}...")
        
        # Performance and configuration insights
        print("\n🔧 Performance Insights:")
        
        # Get performance summary
        performance_summary = quality_monitor.get_performance_summary(days=1)
        if 'total_extractions' in performance_summary:
            print(f"   Recent Extractions: {performance_summary['total_extractions']}")
            if 'success_rate' in performance_summary:
                print(f"   Success Rate: {performance_summary['success_rate']:.1%}")
            if 'average_metrics' in performance_summary:
                avg_metrics = performance_summary['average_metrics']
                print(f"   Avg Quality: {avg_metrics.get('overall_score', 0):.3f}")
                print(f"   Avg Processing Time: {avg_metrics.get('processing_time', 0):.1f}s")
        
        # Configuration insights
        config_report = await config_manager.get_configuration_report()
        if config_report.get('optimization_recommendations'):
            print(f"\n💡 Configuration Recommendations:")
            for rec in config_report['optimization_recommendations'][:3]:  # Show first 3
                print(f"   - {rec}")
        
        print("\n" + "=" * 80)
        print("✅ Enhanced extraction test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Enhanced extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return


async def test_configuration_manager():
    """Test configuration manager functionality"""
    print("\n" + "=" * 60)
    print("Configuration Manager Test")
    print("=" * 60)
    
    try:
        # Test getting different configurations
        configs = {}
        for doc_type in [DocumentType.ACADEMIC_PAPER, DocumentType.TECHNICAL_REPORT]:
            for profile in [ProcessingProfile.FAST, ProcessingProfile.QUALITY]:
                config = await config_manager.get_optimal_config(
                    document_type=doc_type,
                    processing_profile=profile
                )
                configs[f"{doc_type.value}_{profile.value}"] = config
        
        print("✅ Configuration manager test completed")
        print(f"   Generated {len(configs)} different configurations")
        
        # Test configuration report
        report = await config_manager.get_configuration_report()
        print(f"   Configuration report generated: {len(report.get('configurations', {}))} configs")
        
    except Exception as e:
        print(f"❌ Configuration manager test failed: {e}")


async def test_quality_monitor():
    """Test quality monitor functionality"""
    print("\n" + "=" * 60)
    print("Quality Monitor Test")
    print("=" * 60)
    
    try:
        # Test performance summary
        summary = quality_monitor.get_performance_summary(days=7)
        print("✅ Quality monitor test completed")
        print(f"   Performance summary generated: {len(summary)} metrics")
        
        # Test benchmarks
        print(f"   Quality benchmarks available: {len(quality_monitor.benchmarks)} metrics")
        
    except Exception as e:
        print(f"❌ Quality monitor test failed: {e}")


def main():
    """Main entry point"""
    # Parse command line arguments
    pdf_path = None
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
    
    # Run enhanced tests
    async def run_all_tests():
        await test_configuration_manager()
        await test_quality_monitor()
        await test_enhanced_extraction(pdf_path)
    
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()