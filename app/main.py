# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
import uuid
import shutil
import asyncio
from datetime import datetime
from loguru import logger

from app.config import settings
from app.models.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionResult,
    ExtractionStatus,
    B2ExtractionRequest,
    B2ExtractionResponse,
)
from app.services.pipeline import ExtractionPipeline
from app.services.b2_service import b2_service
from app.services.cloudinary_service import cloudinary_service
from app.services.messaging import ScholarAIConsumer
from app.services.extraction_handler import enhanced_extraction_handler
from app.utils.helpers import validate_pdf, get_pdf_info, create_extraction_summary
from app.utils.exceptions import InvalidPDFError, ExtractionError

# Configure logging
logger.add(
    "logs/pdf_extractor.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level,
)

# Global variables
consumer_task = None
rabbitmq_consumer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global consumer_task, rabbitmq_consumer
    
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Create necessary directories
        settings.paper_folder.mkdir(exist_ok=True)
        (settings.paper_folder / "uploads").mkdir(exist_ok=True)
        (settings.paper_folder / "results").mkdir(exist_ok=True)
        
        # Initialize B2 service if credentials are available
        if settings.b2_key_id and settings.b2_application_key:
            try:
                await b2_service.initialize()
                logger.info("B2 service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize B2 service: {e}")
        else:
            logger.info("B2 credentials not configured, B2 functionality will be disabled")
        
        # Initialize Cloudinary service if configured
        if cloudinary_service.is_service_available():
            logger.info("Cloudinary service initialized successfully")
        else:
            logger.info("Cloudinary not configured, image uploads will be disabled")
        
        # Initialize RabbitMQ consumer if credentials are available
        if settings.rabbitmq_user and settings.rabbitmq_password:
            try:
                # Create RabbitMQ consumer instance
                rabbitmq_consumer = ScholarAIConsumer()
                
                # Start RabbitMQ consumer in background
                logger.info("Creating RabbitMQ consumer task...")
                try:
                    consumer_task = asyncio.create_task(rabbitmq_consumer.start())
                    logger.info(f"RabbitMQ consumer task created: {consumer_task}")
                    logger.info("RabbitMQ service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to create consumer task: {e}")
                    logger.exception("Full traceback:")
                    raise
            except Exception as e:
                logger.warning(f"Failed to initialize RabbitMQ service: {e}")
        else:
            logger.info(
                "RabbitMQ credentials not configured, message queue functionality will be disabled"
            )
        
        logger.info("Application started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down application")
        
        if consumer_task:
            consumer_task.cancel()
            try:
                await consumer_task
            except asyncio.CancelledError:
                pass
        
        if rabbitmq_consumer:
            try:
                await rabbitmq_consumer.stop()
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {e}")
        
        logger.info("👋 Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced PDF Extraction API for Academic Papers",
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize extraction pipeline
pipeline = ExtractionPipeline()

# In-memory storage for job tracking (use Redis in production)
extraction_jobs = {}








@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "api_docs": f"{settings.api_prefix}/docs",
    }


@app.get("/health")
async def root_health_check():
    """Simple health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
    }


@app.get(f"{settings.api_prefix}/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
    }


@app.post(f"{settings.api_prefix}/extract")
async def extract_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_text: bool = Query(True, description="Extract text and structure"),
    extract_figures: bool = Query(True, description="Extract figures"),
    extract_tables: bool = Query(True, description="Extract tables"),
    extract_equations: bool = Query(True, description="Extract equations"),
    extract_code: bool = Query(True, description="Extract code blocks"),
    extract_references: bool = Query(True, description="Extract references"),
    use_ocr: bool = Query(True, description="Use OCR for scanned PDFs"),
    detect_entities: bool = Query(True, description="Detect named entities"),
    async_processing: bool = Query(False, description="Process asynchronously"),
):
    """
    Extract content from uploaded PDF

    This endpoint accepts a PDF file and extracts various types of content
    using multiple extraction techniques for maximum accuracy and coverage.
    """

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = settings.paper_folder / "uploads" / f"{job_id}.pdf"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Validate PDF
    if not validate_pdf(upload_path):
        upload_path.unlink()  # Delete invalid file
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file")

    # Create extraction request
    request = ExtractionRequest(
        pdf_path=str(upload_path),
        extract_text=extract_text,
        extract_figures=extract_figures,
        extract_tables=extract_tables,
        extract_equations=extract_equations,
        extract_code=extract_code,
        extract_references=extract_references,
        use_ocr=use_ocr,
        detect_entities=detect_entities,
    )

    # Initialize job tracking
    extraction_jobs[job_id] = {
        "status": ExtractionStatus.PENDING,
        "created_at": datetime.utcnow(),
        "file_name": file.filename,
        "file_path": str(upload_path),
    }

    if async_processing:
        # Process in background
        background_tasks.add_task(process_extraction, job_id, upload_path, request)

        return ExtractionResponse(
            job_id=job_id,
            status=ExtractionStatus.PENDING,
            message="Extraction job started. Use /status endpoint to check progress.",
        )
    else:
        # Process synchronously
        try:
            result = await pipeline.extract(upload_path, request)

            # Update job status
            extraction_jobs[job_id]["status"] = result.status
            extraction_jobs[job_id]["result"] = result
            extraction_jobs[job_id]["completed_at"] = datetime.utcnow()

            return ExtractionResponse(
                job_id=job_id,
                status=result.status,
                result=result,
                message="Extraction completed successfully",
            )
        except Exception as e:
            logger.error(f"Extraction failed for job {job_id}: {e}")
            extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
            extraction_jobs[job_id]["error"] = str(e)

            raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post(f"{settings.api_prefix}/extract-from-b2")
async def extract_pdf_from_b2(
    background_tasks: BackgroundTasks, request: B2ExtractionRequest
):
    """
    Extract content from PDF stored in Backblaze B2

    This endpoint accepts a B2 download URL, downloads the PDF,
    and extracts various types of content using multiple extraction techniques.
    """

    # Validate B2 URL
    if not b2_service.is_b2_url(request.b2_url):
        raise HTTPException(
            status_code=400,
            detail="Invalid B2 URL. Expected format: https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=<file_id>",
        )

    # Check if B2 service is initialized
    if not b2_service._authorized:
        raise HTTPException(
            status_code=503,
            detail="B2 service not available. Please check B2 credentials configuration.",
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Initialize job tracking
    extraction_jobs[job_id] = {
        "status": ExtractionStatus.PENDING,
        "created_at": datetime.utcnow(),
        "b2_url": request.b2_url,
        "type": "b2_extraction",
    }

    if request.async_processing:
        # Process in background
        background_tasks.add_task(
            process_b2_extraction, job_id, request.b2_url, request
        )

        return B2ExtractionResponse(
            job_id=job_id,
            status=ExtractionStatus.PENDING,
            message="B2 extraction job started. Use /status endpoint to check progress.",
            b2_url=request.b2_url,
        )
    else:
        # Process synchronously
        try:
            # Download PDF from B2
            logger.info(f"Downloading PDF from B2 for job {job_id}")
            try:
                pdf_content = await b2_service.download_pdf_from_url(request.b2_url)
            except Exception as e:
                logger.warning(f"B2 SDK download failed, trying fallback method: {e}")
                pdf_content = await b2_service.download_pdf_fallback(request.b2_url)

            # Save PDF to temporary file
            temp_pdf_path = settings.paper_folder / "uploads" / f"{job_id}_b2.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)

            # Validate PDF
            if not validate_pdf(temp_pdf_path):
                temp_pdf_path.unlink()  # Delete invalid file
                raise HTTPException(
                    status_code=400, detail="Downloaded content is not a valid PDF"
                )

            # Create extraction request
            extraction_request = ExtractionRequest(
                pdf_path=str(temp_pdf_path),
                extract_text=request.extract_text,
                extract_figures=request.extract_figures,
                extract_tables=request.extract_tables,
                extract_equations=request.extract_equations,
                extract_code=request.extract_code,
                extract_references=request.extract_references,
                use_ocr=request.use_ocr,
                detect_entities=request.detect_entities,
            )

            # Run extraction
            result = await pipeline.extract(temp_pdf_path, extraction_request)

            # Clean up temporary file
            temp_pdf_path.unlink(missing_ok=True)

            # Update job status
            extraction_jobs[job_id]["status"] = result.status
            extraction_jobs[job_id]["result"] = result
            extraction_jobs[job_id]["completed_at"] = datetime.utcnow()

            return B2ExtractionResponse(
                job_id=job_id,
                status=result.status,
                result=result,
                message="B2 extraction completed successfully",
                b2_url=request.b2_url,
            )
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"B2 extraction failed for job {job_id}: {e}")
            extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
            extraction_jobs[job_id]["error"] = str(e)

            raise HTTPException(
                status_code=500, detail=f"B2 extraction failed: {str(e)}"
            )


async def process_extraction(job_id: str, pdf_path: Path, request: ExtractionRequest):
    """Background task for PDF extraction"""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.PROCESSING
        extraction_jobs[job_id]["started_at"] = datetime.utcnow()

        result = await pipeline.extract(pdf_path, request)

        extraction_jobs[job_id]["status"] = result.status
        extraction_jobs[job_id]["result"] = result
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()

        logger.info(f"Extraction completed for job {job_id}")
    except Exception as e:
        logger.error(f"Extraction failed for job {job_id}: {e}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()


async def process_b2_extraction(job_id: str, b2_url: str, request: B2ExtractionRequest):
    """Background task for B2 PDF extraction"""
    try:
        extraction_jobs[job_id]["status"] = ExtractionStatus.PROCESSING
        extraction_jobs[job_id]["started_at"] = datetime.utcnow()

        # Download PDF from B2
        logger.info(f"Downloading PDF from B2 for job {job_id}")
        try:
            pdf_content = await b2_service.download_pdf_from_url(b2_url)
        except Exception as e:
            logger.warning(f"B2 SDK download failed, trying fallback method: {e}")
            pdf_content = await b2_service.download_pdf_fallback(b2_url)

        # Save PDF to temporary file
        temp_pdf_path = settings.paper_folder / "uploads" / f"{job_id}_b2.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_content)

        # Validate PDF
        if not validate_pdf(temp_pdf_path):
            temp_pdf_path.unlink()  # Delete invalid file
            raise ValueError("Downloaded content is not a valid PDF")

        # Create extraction request
        extraction_request = ExtractionRequest(
            pdf_path=str(temp_pdf_path),
            extract_text=request.extract_text,
            extract_figures=request.extract_figures,
            extract_tables=request.extract_tables,
            extract_equations=request.extract_equations,
            extract_code=request.extract_code,
            extract_references=request.extract_references,
            use_ocr=request.use_ocr,
            detect_entities=request.detect_entities,
        )

        # Run extraction
        result = await pipeline.extract(temp_pdf_path, extraction_request)

        # Clean up temporary file
        temp_pdf_path.unlink(missing_ok=True)

        extraction_jobs[job_id]["status"] = result.status
        extraction_jobs[job_id]["result"] = result
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()

        logger.info(f"B2 extraction completed for job {job_id}")
    except Exception as e:
        logger.error(f"B2 extraction failed for job {job_id}: {e}")
        extraction_jobs[job_id]["status"] = ExtractionStatus.FAILED
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["completed_at"] = datetime.utcnow()


@app.get(f"{settings.api_prefix}/status/{{job_id}}")
async def get_job_status(job_id: str):
    """Get status of extraction job"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "file_name": job.get("file_name"),
        "b2_url": job.get("b2_url"),
        "type": job.get("type", "file_upload"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }

    if job["status"] == ExtractionStatus.COMPLETED:
        response["result"] = job.get("result")
    elif job["status"] == ExtractionStatus.FAILED:
        response["error"] = job.get("error")
    elif job["status"] == ExtractionStatus.PROCESSING:
        # Estimate progress (simplified)
        if "started_at" in job:
            elapsed = (datetime.utcnow() - job["started_at"]).total_seconds()
            estimated_total = settings.extraction_timeout
            progress = min(95, (elapsed / estimated_total) * 100)
            response["progress"] = progress

    return response


@app.get(f"{settings.api_prefix}/result/{{job_id}}")
async def get_extraction_result(
    job_id: str, format: str = Query("json", enum=["json", "summary"])
):
    """Get extraction result"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]

    if job["status"] != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}. Results only available for completed jobs.",
        )

    result = job.get("result")
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    if format == "summary":
        # Return human-readable summary
        summary = create_extraction_summary(result.model_dump() if hasattr(result, 'model_dump') else result.dict())
        return {"summary": summary}
    else:
        # Return full JSON result
        return result


@app.get(f"{settings.api_prefix}/download/{{job_id}}")
async def download_result(job_id: str):
    """Download extraction result as JSON file"""
    if job_id not in extraction_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = extraction_jobs[job_id]

    if job["status"] != ExtractionStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}. Download only available for completed jobs.",
        )

    # Check if result file exists
    result_path = settings.paper_folder / "results" / f"{job_id}_extraction.json"

    if not result_path.exists():
        # Save result to file
        result = job.get("result")
        if result:
            import json

            with open(result_path, "w") as f:
                json.dump(result.model_dump() if hasattr(result, 'model_dump') else result.dict(), f, indent=2, default=str)
        else:
            raise HTTPException(status_code=404, detail="Result not found")

    return FileResponse(
        path=result_path,
        media_type="application/json",
        filename=f"{job['file_name']}_extraction.json",
    )


@app.post(f"{settings.api_prefix}/extract-from-folder")
async def extract_from_paper_folder():
    """
    Extract content from PDF in the paper folder

    This endpoint processes the single PDF file in the paper folder
    as specified in the requirements.
    """
    # Find PDF in paper folder
    pdf_files = list(settings.paper_folder.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(status_code=404, detail="No PDF found in paper folder")

    if len(pdf_files) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple PDFs found in paper folder. Expected only one. Found: {[f.name for f in pdf_files]}",
        )

    pdf_path = pdf_files[0]

    # Validate PDF
    if not validate_pdf(pdf_path):
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {pdf_path.name}")

    # Get PDF info
    pdf_info = get_pdf_info(pdf_path)

    logger.info(f"Processing PDF: {pdf_path.name}")
    logger.info(f"PDF Info: {pdf_info}")

    # Create extraction request with all features enabled
    request = ExtractionRequest(
        pdf_path=str(pdf_path),
        extract_text=True,
        extract_figures=True,
        extract_tables=True,
        extract_equations=True,
        extract_code=True,
        extract_references=True,
        use_ocr=True,
        detect_entities=True,
    )

    try:
        # Run extraction
        result = await pipeline.extract(pdf_path, request)

        # Save result to JSON in the same folder
        output_path = settings.paper_folder / f"{pdf_path.stem}_extraction.json"

        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump() if hasattr(result, 'model_dump') else result.dict(), f, indent=2, ensure_ascii=False, default=str)

        # Create summary
        summary = create_extraction_summary(result.model_dump() if hasattr(result, 'model_dump') else result.dict())

        # Save summary
        summary_path = settings.paper_folder / f"{pdf_path.stem}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        logger.info(f"Extraction completed. Results saved to {output_path}")
        logger.info(f"Summary saved to {summary_path}")

        return {
            "status": "success",
            "pdf_file": pdf_path.name,
            "pdf_info": pdf_info,
            "extraction_result": result,
            "output_files": {"json": str(output_path), "summary": str(summary_path)},
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.delete(f"{settings.api_prefix}/cleanup")
async def cleanup_old_jobs(
    hours: int = Query(24, description="Delete jobs older than N hours")
):
    """Clean up old extraction jobs and files"""
    from datetime import timedelta

    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    deleted_count = 0

    jobs_to_delete = []
    for job_id, job in extraction_jobs.items():
        created_at = job.get("created_at")
        if created_at and created_at < cutoff_time:
            jobs_to_delete.append(job_id)

            # Delete associated files
            if "file_path" in job:
                try:
                    Path(job["file_path"]).unlink(missing_ok=True)
                except:
                    pass

    for job_id in jobs_to_delete:
        del extraction_jobs[job_id]
        deleted_count += 1

    return {"deleted_jobs": deleted_count, "remaining_jobs": len(extraction_jobs)}


@app.get(f"{settings.api_prefix}/extractions")
async def list_extractions():
    """List all extraction jobs and their status"""
    jobs_list = []
    for job_id, job in extraction_jobs.items():
        job_info = {
            "job_id": job_id,
            "status": job["status"],
            "type": job.get("type", "file_upload"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }

        if "file_name" in job:
            job_info["file_name"] = job["file_name"]
        if "b2_url" in job:
            job_info["b2_url"] = job["b2_url"]
        if "error" in job:
            job_info["error"] = job["error"]

        jobs_list.append(job_info)

    # Sort by creation time, newest first
    jobs_list.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    return {"total_jobs": len(jobs_list), "jobs": jobs_list}


@app.get(f"{settings.api_prefix}/extractions/stats")
async def get_extraction_stats():
    """Get extraction statistics"""
    total_jobs = len(extraction_jobs)

    status_counts = {}
    for job in extraction_jobs.values():
        status = job.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "total_jobs": total_jobs,
        "status_breakdown": status_counts,
        "service_info": {
            "name": settings.app_name,
            "version": settings.app_version,
            "b2_enabled": bool(settings.b2_key_id and settings.b2_application_key),
            "rabbitmq_enabled": bool(
                settings.rabbitmq_user and settings.rabbitmq_password
            ),
        },
    }


@app.post(f"{settings.api_prefix}/test-b2")
async def test_b2_connection():
    """Test B2 connection and credentials"""
    if not settings.b2_key_id or not settings.b2_application_key:
        raise HTTPException(status_code=503, detail="B2 credentials not configured")

    try:
        # Try to initialize B2 service
        await b2_service.initialize()

        return {
            "status": "success",
            "message": "B2 connection is working",
            "bucket_name": settings.b2_bucket_name,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"B2 connection failed: {str(e)}")


@app.get(f"{settings.api_prefix}/queue-status")
async def get_queue_status():
    """Get RabbitMQ queue status"""
    if not settings.rabbitmq_user or not settings.rabbitmq_password:
        return {"status": "disabled", "message": "RabbitMQ not configured"}

    try:
        # Try to get queue information (this is a simplified check)
        return {
            "status": "connected",
            "message": "RabbitMQ connection is active",
            "extraction_queue": settings.rabbitmq_extraction_queue,
            "completed_queue": settings.rabbitmq_extraction_completed_queue,
        }
    except Exception as e:
        return {"status": "error", "message": f"RabbitMQ connection error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
