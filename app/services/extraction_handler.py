"""
Handler for processing extraction requests from RabbitMQ.
"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from app.config import settings
from app.services.pipeline import ExtractionPipeline
from app.services.b2_service import b2_service
from app.services.cloudinary_service import cloudinary_service
from app.services.local_storage_service import local_storage_service
from app.models.schemas import ExtractionRequest, ExtractionResult
from app.models.enums import ExtractionStatus
from app.utils.helpers import validate_pdf
from app.utils.exceptions import ExtractionError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    """Context for processing an extraction request"""

    job_id: str
    paper_id: str
    correlation_id: str
    b2_url: str
    start_time: datetime
    retry_count: int = 0
    quality_target: float = 0.7
    processing_hints: Dict[str, Any] = None
    result: Optional[Any] = None
    status: str = "PROCESSING"
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.processing_hints is None:
            self.processing_hints = {}


@dataclass
class AdaptiveConfig:
    """Adaptive configuration based on document characteristics and performance"""

    # Quality thresholds
    min_acceptable_quality: float = 0.6
    target_quality: float = 0.8
    max_processing_time: int = 600  # 10 minutes

    # Method selection weights
    method_weights: Dict[str, float] = None

    # Retry configuration
    max_retries: int = 2
    retry_delay: int = 30

    # Performance targets
    target_processing_speed: float = 120.0  # seconds per document

    def __post_init__(self):
        if self.method_weights is None:
            self.method_weights = {
                "grobid": 0.9,
                "table_transformer": 0.8,
        
                "cv_detection": 0.6,
                "ocr_fallback": 0.5,
            }


class EnhancedExtractionHandler:
    """
    Enhanced handler with adaptive processing, quality monitoring,
    and intelligent error recovery
    """

    def __init__(self):
        self.pipeline = ExtractionPipeline()
        self.adaptive_config = AdaptiveConfig()
        self.processing_history = {}

        # Performance tracking
        self.recent_performance = []
        self.method_success_rates = {}
        self.quality_trends = []

        # Circuit breaker states for methods
        self.method_circuit_breakers = {
            "grobid": {"failures": 0, "last_failure": None, "is_open": False},
            "table_transformer": {
                "failures": 0,
                "last_failure": None,
                "is_open": False,
            },
    
        }

        # Initialize lazily (will be called when needed)
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure handler is initialized"""
        if not self._initialized:
            await self._load_performance_history()
            self._initialized = True

    async def _load_performance_history(self):
        """Load historical performance data for adaptive processing"""
        try:
            if settings.store_locally:
                # Local storage mode: load from file
                history_file = settings.paper_folder / "extraction_performance.json"
                if history_file.exists():
                    with open(history_file, "r") as f:
                        data = json.load(f)
                        self.recent_performance = data.get("recent_performance", [])
                        self.method_success_rates = data.get("method_success_rates", {})
                        logger.info("Loaded extraction performance history from local file")
            else:
                # Cloud-only mode: try to load from Cloudinary
                try:
                    # For now, start with empty data when in cloud-only mode
                    # In a production system, you might want to store the Cloudinary URL
                    # and load from there, but for simplicity, we'll start fresh
                    self.recent_performance = []
                    self.method_success_rates = {}
                    logger.info("Starting with empty performance history in cloud-only mode")
                except Exception as e:
                    logger.warning(f"Failed to load performance history from Cloudinary: {e}")
                    # Fallback to empty data
                    self.recent_performance = []
                    self.method_success_rates = {}
                    
        except Exception as e:
            logger.warning(f"Failed to load performance history: {e}")
            # Fallback to empty data
            self.recent_performance = []
            self.method_success_rates = {}

    async def _save_performance_history(self):
        """Save performance history based on STORE_LOCALLY setting"""
        try:
            data = {
                "recent_performance": self.recent_performance[-100:],  # Keep last 100
                "method_success_rates": self.method_success_rates,
                "last_updated": datetime.utcnow().isoformat(),
            }
            
            if settings.store_locally:
                # Local storage mode: save to file
                history_file = settings.paper_folder / "extraction_performance.json"
                with open(history_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                logger.debug("Performance history saved locally")
            else:
                # Cloud-only mode: do not save or upload anything
                logger.debug("STORE_LOCALLY=False: Skipping performance history storage (no local files, no Cloudinary upload)")
                    
        except Exception as e:
            logger.warning(f"Failed to save performance history: {e}")

    async def handle_extraction_request(self, message: Dict[str, Any]) -> bool:
        """
        Handle extraction request with enhanced error handling and validation
        """
        try:
            await self._ensure_initialized()

            logger.info(f"Processing extraction request: {message.get('jobId')}")

            # Validate request
            validation_result = self._validate_request(message)
            if not validation_result.is_valid:
                logger.error(f"Invalid request: {validation_result.error}")
                return False

            # Create processing context
            context = ProcessingContext(
                job_id=message.get("jobId"),
                paper_id=message.get("paperId"),
                correlation_id=message.get("correlationId"),
                b2_url=message.get("b2Url"),
                start_time=datetime.utcnow(),
                quality_target=message.get("qualityTarget", 0.7),
                processing_hints=message.get("processingHints", {}),
            )

            # Store context in history
            self.processing_history[context.job_id] = context

            # Process extraction asynchronously with context
            asyncio.create_task(
                self._process_extraction_with_adaptation(message, context)
            )

            return True

        except Exception as e:
            logger.error(f"Error handling extraction request: {e}")
            return False

    def _validate_request(self, message: Dict[str, Any]) -> "ValidationResult":
        """Validate extraction request"""
        required_fields = ["jobId", "paperId", "correlationId", "b2Url"]

        for field in required_fields:
            if not message.get(field):
                return ValidationResult(False, f"Missing required field: {field}")

        # Validate quality target
        quality_target = message.get("qualityTarget", 0.7)
        if not 0.1 <= quality_target <= 1.0:
            return ValidationResult(False, f"Invalid quality target: {quality_target}")

        return ValidationResult(True, None)

    async def _process_extraction_with_adaptation(
        self, message: Dict[str, Any], context: ProcessingContext
    ):
        """Process extraction with adaptive configuration and quality monitoring"""
        try:
            # Create adaptive request based on message and context
            adaptive_request = await self._create_adaptive_request(
                message, {}, context
            )

            # Download and validate PDF
            pdf_path = await self._download_and_validate_pdf(context)

            while context.retry_count <= self.adaptive_config.max_retries:
                try:
                    # Run extraction with adaptive configuration
                    logger.info(
                        f"Starting extraction attempt {context.retry_count + 1} for {context.job_id}"
                    )

                    result = await self._run_adaptive_extraction(
                        pdf_path, adaptive_request, context
                    )

                    # Evaluate quality using pipeline's quality metrics
                    quality_metrics = result.quality_metrics or {}
                    overall_score = quality_metrics.get("overall_score", 0.0)

                    # Check if quality meets requirements
                    if overall_score >= context.quality_target:
                        logger.info(f"Quality target met: {overall_score:.2f}")
                        await self._handle_successful_extraction(
                            result, context, quality_metrics
                        )
                        return
                    else:
                        logger.warning(
                            f"Quality below target: {overall_score:.2f} < {context.quality_target}"
                        )

                        # Decide whether to retry
                        if await self._should_retry_extraction(
                            result, quality_metrics, context
                        ):
                            context.retry_count += 1
                            adaptive_request = await self._adapt_for_retry(
                                adaptive_request, result, quality_metrics, context
                            )
                            logger.info(f"Retrying with adapted configuration")
                            continue
                        else:
                            # Accept partial result
                            logger.info("Accepting partial result")
                            await self._handle_partial_extraction(
                                result, context, quality_metrics
                            )
                            return

                except ExtractionError as e:
                    logger.error(
                        f"Extraction failed (attempt {context.retry_count + 1}): {e}"
                    )

                    if context.retry_count < self.adaptive_config.max_retries:
                        context.retry_count += 1
                        # Adapt for error recovery
                        adaptive_request = await self._adapt_for_error_recovery(
                            adaptive_request, str(e), context
                        )
                        await asyncio.sleep(self.adaptive_config.retry_delay)
                        continue
                    else:
                        raise e

            # Max retries reached
            raise ExtractionError("Maximum retry attempts reached")

        except Exception as e:
            logger.error(f"Extraction failed for job {context.job_id}: {e}")
            await self._handle_extraction_failure(context, str(e))
        finally:
            # Cleanup
            if "pdf_path" in locals() and pdf_path.exists():
                pdf_path.unlink(missing_ok=True)



    async def _create_adaptive_request(
        self,
        message: Dict[str, Any],
        doc_analysis: Dict[str, Any],
        context: ProcessingContext,
    ) -> ExtractionRequest:
        """Create adaptive extraction request based on performance history"""

        # Base request from message
        request = ExtractionRequest(
            extract_text=message.get("extractText", True),
            extract_figures=message.get("extractFigures", True),
            extract_tables=message.get("extractTables", True),
            extract_equations=message.get("extractEquations", True),
            extract_code=message.get("extractCode", True),
            extract_references=message.get("extractReferences", True),
            use_ocr=message.get("useOcr", True),
            detect_entities=message.get("detectEntities", True),
        )

        # Adapt based on performance history
        if self.recent_performance:
            avg_quality = sum(
                p["quality"] for p in self.recent_performance[-10:]
            ) / min(len(self.recent_performance), 10)

            if avg_quality < 0.6:  # Recent poor performance
                # Enable more methods
                request.use_ocr = True
                request.detect_entities = True
                logger.info(
                    "Enabling additional methods due to recent poor performance"
                )

        # Circuit breaker adaptations
        if self._is_method_circuit_open("grobid"):
            logger.warning("GROBID circuit breaker is open, falling back to OCR")
            request.use_ocr = True

        # Adapt based on processing hints
        hints = context.processing_hints
        if hints.get("fast_mode"):
            request.use_ocr = False
            request.detect_entities = False
        elif hints.get("high_quality_mode"):
            request.use_ocr = True
            request.detect_entities = True
            request.timeout = self.adaptive_config.max_processing_time * 2

        return request

    def _is_method_circuit_open(self, method_name: str) -> bool:
        """Check if circuit breaker is open for a method"""
        circuit = self.method_circuit_breakers.get(method_name, {})

        if not circuit.get("is_open", False):
            return False

        # Check if enough time has passed to try again
        last_failure = circuit.get("last_failure")
        if last_failure:
            time_since_failure = datetime.utcnow() - last_failure
            if time_since_failure > timedelta(minutes=10):  # Try again after 10 minutes
                circuit["is_open"] = False
                circuit["failures"] = 0
                return False

        return True

    async def _download_and_validate_pdf(self, context: ProcessingContext) -> Path:
        """Download and validate PDF with enhanced error handling"""
        temp_pdf_path = settings.paper_folder / "uploads" / f"{context.job_id}_b2.pdf"
        temp_pdf_path.parent.mkdir(exist_ok=True)

        try:
            # Download with retries
            pdf_content = await self._download_with_retries(
                context.b2_url, max_retries=3
            )

            # Save to temporary file
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_content)

            # Enhanced validation
            validation_result = await self._enhanced_pdf_validation(temp_pdf_path)
            if not validation_result.is_valid:
                temp_pdf_path.unlink(missing_ok=True)
                raise ExtractionError(
                    f"PDF validation failed: {validation_result.error}"
                )

            return temp_pdf_path

        except Exception as e:
            if temp_pdf_path.exists():
                temp_pdf_path.unlink(missing_ok=True)
            raise ExtractionError(f"PDF download/validation failed: {str(e)}")

    async def _download_with_retries(self, b2_url: str, max_retries: int = 3) -> bytes:
        """Download PDF with exponential backoff retries"""
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    # Try SDK first
                    return await b2_service.download_pdf_from_url(b2_url)
                else:
                    # Use fallback method
                    return await b2_service.download_pdf_fallback(b2_url)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                wait_time = (2**attempt) * 1.0  # Exponential backoff
                logger.warning(
                    f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

    async def _enhanced_pdf_validation(self, pdf_path: Path) -> "ValidationResult":
        """Enhanced PDF validation with detailed checks"""
        try:
            # Basic validation
            if not validate_pdf(pdf_path):
                return ValidationResult(False, "Basic PDF validation failed")

            # Size validation
            file_size = pdf_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                return ValidationResult(False, "PDF file too small")
            if file_size > 100 * 1024 * 1024:  # More than 100MB
                return ValidationResult(False, "PDF file too large")

            # Content validation using PyMuPDF
            import fitz

            try:
                doc = fitz.open(str(pdf_path))

                # Check page count
                if doc.page_count == 0:
                    doc.close()
                    return ValidationResult(False, "PDF has no pages")

                if doc.page_count > 1000:  # Sanity check
                    doc.close()
                    return ValidationResult(False, "PDF has too many pages")

                # Check if pages have content
                has_content = False
                for page_num in range(min(3, doc.page_count)):  # Check first 3 pages
                    page = doc[page_num]
                    if page.get_text().strip():
                        has_content = True
                        break

                doc.close()

                if not has_content:
                    return ValidationResult(
                        False, "PDF appears to have no text content"
                    )

            except Exception as e:
                return ValidationResult(
                    False, f"PDF structure validation failed: {str(e)}"
                )

            return ValidationResult(True, None)

        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")

    async def _run_adaptive_extraction(
        self, pdf_path: Path, request: ExtractionRequest, context: ProcessingContext
    ) -> ExtractionResult:
        """Run extraction with adaptive method selection"""

        # Update extraction methods based on circuit breakers
        available_methods = []

        if not self._is_method_circuit_open("grobid") and request.extract_text:
            available_methods.append("grobid")

        if (
            not self._is_method_circuit_open("table_transformer")
            and request.extract_tables
        ):
            available_methods.append("table_transformer")



        # Run extraction
        try:
            result = await self.pipeline.extract(pdf_path, request, skip_local_storage=True)

            # Update method success tracking
            for method in result.extraction_methods:
                self._record_method_success(method)

            return result

        except Exception as e:
            # Update method failure tracking
            for method in available_methods:
                self._record_method_failure(method)

            raise e

    def _record_method_success(self, method_name: str):
        """Record successful method execution"""
        if method_name in self.method_circuit_breakers:
            circuit = self.method_circuit_breakers[method_name]
            circuit["failures"] = 0
            circuit["is_open"] = False

        # Update success rates
        if method_name not in self.method_success_rates:
            self.method_success_rates[method_name] = {"success": 0, "total": 0}

        self.method_success_rates[method_name]["success"] += 1
        self.method_success_rates[method_name]["total"] += 1

    def _record_method_failure(self, method_name: str):
        """Record failed method execution and update circuit breaker"""
        if method_name in self.method_circuit_breakers:
            circuit = self.method_circuit_breakers[method_name]
            circuit["failures"] += 1
            circuit["last_failure"] = datetime.utcnow()

            # Open circuit breaker if too many failures
            if circuit["failures"] >= 3:
                circuit["is_open"] = True
                logger.warning(f"Circuit breaker opened for {method_name}")

        # Update success rates
        if method_name not in self.method_success_rates:
            self.method_success_rates[method_name] = {"success": 0, "total": 0}

        self.method_success_rates[method_name]["total"] += 1

    async def _should_retry_extraction(
        self,
        result: ExtractionResult,
        quality_metrics: Dict[str, Any],
        context: ProcessingContext,
    ) -> bool:
        """Decide whether to retry extraction based on quality and context"""

        # Don't retry if already at max retries
        if context.retry_count >= self.adaptive_config.max_retries:
            return False

        # Don't retry if quality is close to target (within 0.1)
        overall_score = quality_metrics.get("overall_score", 0.0)
        if overall_score >= context.quality_target - 0.1:
            return False

        # Don't retry if processing time is already too high
        processing_time = (datetime.utcnow() - context.start_time).total_seconds()
        if processing_time > self.adaptive_config.max_processing_time * 0.8:
            return False

        # Retry if specific issues can be addressed
        # Retry if table false positive rate is high (can be improved)
        table_false_positive_rate = quality_metrics.get(
            "table_false_positive_rate", 0.0
        )
        if table_false_positive_rate > 0.4:
            return True

        # Retry if text coherence is very low (OCR might help)
        text_coherence = quality_metrics.get("text_coherence", 0.0)
        if text_coherence < 0.4 and "nougat_tesseract" not in result.extraction_methods:
            return True

        # Retry if coverage is very low
        extraction_coverage = quality_metrics.get("extraction_coverage", 0.0)
        if extraction_coverage < 0.4:
            return True

        return False

    async def _adapt_for_retry(
        self,
        request: ExtractionRequest,
        result: ExtractionResult,
        quality_metrics: Dict[str, Any],
        context: ProcessingContext,
    ) -> ExtractionRequest:
        """Adapt extraction request for retry based on previous results"""

        adapted_request = request.copy()

        # If table accuracy is poor, adjust table extraction
        table_false_positive_rate = quality_metrics.get(
            "table_false_positive_rate", 0.0
        )
        if table_false_positive_rate > 0.4:
            logger.info("Adapting for better table extraction")
            # Could add specific table extraction parameters here

        # If text coherence is poor, try OCR
        text_coherence = quality_metrics.get("text_coherence", 0.0)
        if text_coherence < 0.4 and "nougat_tesseract" not in result.extraction_methods:
            logger.info("Enabling OCR for better text extraction")
            adapted_request.use_ocr = True

        # If coverage is low, enable more methods
        extraction_coverage = quality_metrics.get("extraction_coverage", 0.0)
        if extraction_coverage < 0.5:
            logger.info("Enabling additional extraction methods")
            adapted_request.use_ocr = True
            adapted_request.detect_entities = True

        # Increase timeout for retry
        adapted_request.timeout = min(
            self.adaptive_config.max_processing_time,
            (adapted_request.timeout or 300) * 1.5,
        )

        return adapted_request

    async def _adapt_for_error_recovery(
        self, request: ExtractionRequest, error_message: str, context: ProcessingContext
    ) -> ExtractionRequest:
        """Adapt request for error recovery"""

        adapted_request = request.copy()

        # If GROBID failed, fall back to OCR
        if "grobid" in error_message.lower():
            logger.info("GROBID failed, falling back to OCR")
            adapted_request.use_ocr = True
            self._record_method_failure("grobid")

        # If table extraction failed, simplify
        if "table" in error_message.lower():
            logger.info("Table extraction failed, using fallback methods")
            self._record_method_failure("table_transformer")

        # If figure extraction failed, try different methods
        if "figure" in error_message.lower():
            logger.info("Figure extraction failed, using fallback methods")
    

        # General timeout errors - reduce scope
        if "timeout" in error_message.lower():
            logger.info("Timeout occurred, reducing extraction scope")
            adapted_request.detect_entities = False
            adapted_request.use_ocr = False

        return adapted_request

    async def _handle_successful_extraction(
        self,
        result: ExtractionResult,
        context: ProcessingContext,
        quality_metrics: Dict[str, Any],
    ):
        """Handle successful extraction with quality monitoring"""

        # Store result in context
        context.result = result
        context.status = "COMPLETED"
        context.completed_at = datetime.utcnow()

        # Record performance metrics
        processing_time = (datetime.utcnow() - context.start_time).total_seconds()
        overall_score = quality_metrics.get("overall_score", 0.0)

        performance_record = {
            "job_id": context.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "quality": overall_score,
            "retry_count": context.retry_count,
            "methods_used": result.extraction_methods,
            "success": True,
        }

        self.recent_performance.append(performance_record)
        if len(self.recent_performance) > 100:
            self.recent_performance.pop(0)

        # Save performance history
        await self._save_performance_history()

        # Store locally if enabled
        if settings.store_locally:
            try:
                stored_paths = await local_storage_service.store_extraction_result(
                    result, context.job_id, context.paper_id
                )
                logger.info(f"Stored extraction result locally: {stored_paths}")
            except Exception as e:
                logger.error(f"Failed to store extraction result locally: {e}")
        
        # Send completion message
        await self._send_completion_message(
            job_id=context.job_id,
            paper_id=context.paper_id,
            correlation_id=context.correlation_id,
            status="completed",
            result=result,
            message="Extraction completed successfully",
            quality_metrics=quality_metrics,
        )

        logger.info(
            f"Successfully completed extraction for job {context.job_id} "
            f"in {processing_time:.1f}s with quality {overall_score:.2f}"
        )

    async def _handle_partial_extraction(
        self,
        result: ExtractionResult,
        context: ProcessingContext,
        quality_metrics: Dict[str, Any],
    ):
        """Handle partial extraction (below target quality but acceptable)"""

        # Store result in context
        context.result = result
        context.status = "PARTIAL"
        context.completed_at = datetime.utcnow()

        processing_time = (datetime.utcnow() - context.start_time).total_seconds()
        overall_score = quality_metrics.get("overall_score", 0.0)

        performance_record = {
            "job_id": context.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "quality": overall_score,
            "retry_count": context.retry_count,
            "methods_used": result.extraction_methods,
            "success": True,
            "partial": True,
        }

        self.recent_performance.append(performance_record)

        # Store locally if enabled
        if settings.store_locally:
            try:
                stored_paths = await local_storage_service.store_extraction_result(
                    result, context.job_id, context.paper_id
                )
                logger.info(f"Stored partial extraction result locally: {stored_paths}")
            except Exception as e:
                logger.error(f"Failed to store partial extraction result locally: {e}")

        await self._send_completion_message(
            job_id=context.job_id,
            paper_id=context.paper_id,
            correlation_id=context.correlation_id,
            status="partial",
            result=result,
            message=f"Extraction completed with quality {overall_score:.2f} "
            f"(below target {context.quality_target:.2f})",
            quality_metrics=quality_metrics,
        )

        logger.info(
            f"Completed partial extraction for job {context.job_id} "
            f"with quality {overall_score:.2f}"
        )

    async def _handle_extraction_failure(
        self, context: ProcessingContext, error_message: str
    ):
        """Handle extraction failure with detailed reporting"""

        # Store error status in context
        context.status = "FAILED"
        context.completed_at = datetime.utcnow()

        processing_time = (datetime.utcnow() - context.start_time).total_seconds()

        performance_record = {
            "job_id": context.job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "quality": 0.0,
            "retry_count": context.retry_count,
            "methods_used": [],
            "success": False,
            "error": error_message,
        }

        self.recent_performance.append(performance_record)

        await self._send_completion_message(
            job_id=context.job_id,
            paper_id=context.paper_id,
            correlation_id=context.correlation_id,
            status="failed",
            result=None,
            message=f"Extraction failed after {context.retry_count + 1} attempts: {error_message}",
        )

    async def _send_completion_message(
        self,
        job_id: str,
        paper_id: str,
        correlation_id: str,
        status: str,
        result: Optional[ExtractionResult] = None,
        message: str = "",
        quality_metrics: Dict[str, Any] = None,
    ):
        """Enhanced completion message with quality metrics"""
        try:
            completion_message = {
                "jobId": job_id,
                "paperId": paper_id,
                "correlationId": correlation_id,
                "status": status,
                "message": message,
                "completedAt": datetime.utcnow().isoformat(),
            }

            if result and status in ["completed", "partial"]:
                # Convert result to dictionary
                if hasattr(result, "dict"):
                    result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict()
                else:
                    result_dict = result

                # Serialize datetime objects before JSON conversion
                result_dict = self._serialize_datetime_objects(result_dict)

                completion_message.update(
                    {
                        "extractionResult": json.dumps(result_dict),
                        "processingTime": getattr(result, "processing_time", None),
                        "extractionCoverage": getattr(
                            result, "extraction_coverage", None
                        ),
                        "confidenceScores": json.dumps(getattr(result, "confidence_scores", {})),
                        "errors": json.dumps(getattr(result, "errors", [])),
                        "warnings": json.dumps(getattr(result, "warnings", [])),
                    }
                )

                # Add quality metrics if available
                if quality_metrics:
                    completion_message["qualityMetrics"] = {
                        "overallScore": quality_metrics.get("overall_score", 0.0),
                        "extractionCoverage": quality_metrics.get(
                            "extraction_coverage", 0.0
                        ),
                        "textCoherence": quality_metrics.get("text_coherence", 0.0),
                        "structurePreservation": quality_metrics.get(
                            "structure_preservation", 0.0
                        ),
                        "referenceAccuracy": quality_metrics.get(
                            "reference_accuracy", 0.0
                        ),
                        "contentTypeAccuracy": quality_metrics.get(
                            "content_type_accuracy", 0.0
                        ),
                    }

            # Store result in context for later retrieval
            if job_id in self.processing_history:
                context = self.processing_history[job_id]
                context.result = result
                # Don't overwrite the context status - it's already set correctly
                # context.status = status  # Remove this line
                context.completed_at = datetime.utcnow()

            # Note: The actual message publishing is now handled by the messaging handler
            # This method just prepares the completion data
            logger.info(f"Prepared completion message for job {job_id}")

        except Exception as e:
            logger.error(f"Error preparing completion message for job {job_id}: {e}")

    async def get_extraction_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get extraction result for a specific job ID.
        
        Args:
            job_id: The job ID to get results for
            
        Returns:
            Dict containing extraction result or None if not found
        """
        try:
            # Check if we have the result in processing history
            if job_id in self.processing_history:
                context = self.processing_history[job_id]
                if hasattr(context, 'result') and context.result:
                    # Convert result to dict format expected by RabbitMQ
                    # Handle datetime serialization properly
                    if hasattr(context.result, 'dict'):
                        result_dict_raw = context.result.model_dump() if hasattr(context.result, 'model_dump') else context.result.dict()
                        # Convert datetime objects to ISO format strings
                        result_dict = self._serialize_datetime_objects(result_dict_raw)
                    else:
                        result_dict = str(context.result)
                    
                    return {
                        "jobId": job_id,
                        "paperId": context.paper_id,
                        "correlationId": context.correlation_id,
                        "status": context.status if hasattr(context, 'status') else "COMPLETED",
                        "completedAt": (context.completed_at or context.start_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                        "extractionCoverage": context.quality_target,
                        "extractionResult": json.dumps(result_dict),
                        "message": "Extraction completed successfully"
                    }
                
                # If not found in history, check if it's still processing
                if hasattr(context, 'status') and context.status == "PROCESSING":
                    return {
                        "jobId": job_id,
                        "paperId": context.paper_id,
                        "correlationId": context.correlation_id,
                        "status": "PROCESSING",
                        "completedAt": None,
                        "extractionCoverage": None,
                        "extractionResult": None,
                        "message": "Extraction is still in progress"
                    }
            
            # Job not found
            return {
                "jobId": job_id,
                "status": ExtractionStatus.NOT_FOUND,
                "completedAt": None,
                "extractionCoverage": None,
                "extractionResult": None,
                "message": "Job not found"
            }
            
        except Exception as e:
            logger.error(f"Error getting extraction result for job {job_id}: {e}")
            return {
                "jobId": job_id,
                "status": "ERROR",
                "completedAt": None,
                "extractionCoverage": None,
                "extractionResult": None,
                "message": f"Error retrieving result: {str(e)}"
            }

    def _serialize_datetime_objects(self, obj: Any) -> Any:
        """
        Recursively serialize datetime objects to ISO format strings.
        
        Args:
            obj: Object that may contain datetime objects
            
        Returns:
            Object with datetime objects converted to ISO strings
        """
        if isinstance(obj, dict):
            return {key: self._serialize_datetime_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
            return obj.isoformat()
        else:
            return obj

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary for monitoring"""
        if not self.recent_performance:
            return {"message": "No performance data available"}

        recent_extractions = self.recent_performance[-20:]  # Last 20 extractions

        summary = {
            "total_extractions": len(self.recent_performance),
            "recent_extractions": len(recent_extractions),
            "success_rate": sum(1 for p in recent_extractions if p["success"])
            / len(recent_extractions),
            "average_quality": sum(p["quality"] for p in recent_extractions)
            / len(recent_extractions),
            "average_processing_time": sum(
                p["processing_time"] for p in recent_extractions
            )
            / len(recent_extractions),
            "method_success_rates": self.method_success_rates,
            "circuit_breaker_status": {
                method: circuit["is_open"]
                for method, circuit in self.method_circuit_breakers.items()
            },
            "recent_trends": self._calculate_recent_trends(),
            "recommendations": self._generate_performance_recommendations(),
        }

        return summary

    def _calculate_recent_trends(self) -> Dict[str, str]:
        """Calculate recent performance trends"""
        if len(self.recent_performance) < 10:
            return {"message": "Insufficient data for trend analysis"}

        recent_quality = [p["quality"] for p in self.recent_performance[-10:]]
        recent_time = [p["processing_time"] for p in self.recent_performance[-10:]]

        import numpy as np

        # Quality trend
        quality_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        time_trend = np.polyfit(range(len(recent_time)), recent_time, 1)[0]

        trends = {}

        if quality_trend > 0.01:
            trends["quality"] = "Improving"
        elif quality_trend < -0.01:
            trends["quality"] = "Declining"
        else:
            trends["quality"] = "Stable"

        if time_trend > 5.0:
            trends["processing_speed"] = "Slowing"
        elif time_trend < -5.0:
            trends["processing_speed"] = "Improving"
        else:
            trends["processing_speed"] = "Stable"

        return trends

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        if not self.recent_performance:
            return recommendations

        recent_extractions = self.recent_performance[-20:]

        # Check success rate
        success_rate = sum(1 for p in recent_extractions if p["success"]) / len(
            recent_extractions
        )
        if success_rate < 0.8:
            recommendations.append(
                f"Success rate is low ({success_rate:.1%}). Consider reviewing error patterns and adjusting method selection."
            )

        # Check quality
        avg_quality = sum(p["quality"] for p in recent_extractions) / len(
            recent_extractions
        )
        if avg_quality < 0.7:
            recommendations.append(
                f"Average quality is low ({avg_quality:.2f}). Consider enabling additional extraction methods."
            )

        # Check processing time
        avg_time = sum(p["processing_time"] for p in recent_extractions) / len(
            recent_extractions
        )
        if avg_time > 300:  # 5 minutes
            recommendations.append(
                f"Processing time is high ({avg_time:.1f}s). Consider optimizing extraction pipeline."
            )

        # Check circuit breakers
        open_circuits = [
            method
            for method, circuit in self.method_circuit_breakers.items()
            if circuit["is_open"]
        ]
        if open_circuits:
            recommendations.append(
                f"Circuit breakers are open for: {', '.join(open_circuits)}. Check service health."
            )

        return recommendations


@dataclass
class ValidationResult:
    """Result of request validation"""

    is_valid: bool
    error: Optional[str] = None


# Global handler instance
enhanced_extraction_handler = EnhancedExtractionHandler()
