"""
Message Handlers for ScholarAI Extractor

Handles different types of messages and routes them to appropriate
processing logic for extraction tasks.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from aio_pika import IncomingMessage

from .base_handler import BaseMessageHandler
from ..extraction_handler import enhanced_extraction_handler

logger = logging.getLogger(__name__)


class ExtractionMessageHandler(BaseMessageHandler):
    """
    Handles extraction messages and processes them using the enhanced extraction handler.
    
    Responsible for:
    - Message validation
    - Extraction orchestration
    - Result publishing
    - Error handling
    """

    def __init__(self):
        super().__init__()
        self.connection_manager = None

    async def initialize(self, connection_manager):
        """Initialize the handler with connection manager."""
        self.connection_manager = connection_manager
        logger.info("✅ Extraction message handler initialized")

    async def handle_message(self, message: IncomingMessage) -> bool:
        """
        Handle extraction message processing.
        
        Args:
            message: Incoming RabbitMQ message
            
        Returns:
            bool: True if message processed successfully
        """
        try:
            # Parse message
            message_data = await self._parse_message(message)
            if not message_data:
                return False
            
            # Validate message
            if not self._validate_extraction_message(message_data):
                logger.error("❌ Invalid extraction message format")
                await message.reject(requeue=False)
                return False
            
            job_id = message_data.get('jobId', 'unknown')
            logger.info(f"📄 Processing extraction request: {job_id}")
            
            # Process extraction request using enhanced extraction handler
            success = await enhanced_extraction_handler.handle_extraction_request(message_data)
            
            if success:
                # Wait for extraction to complete with polling
                extraction_result = await self._wait_for_extraction_completion(job_id)
                
                if extraction_result and extraction_result.get('status') in ['COMPLETED', 'PARTIAL']:
                    # Publish result
                    if self.connection_manager:
                        result_success = await self.connection_manager.publish_extraction_result(extraction_result)
                        if result_success:
                            logger.info(f"✅ Extraction result published for: {job_id}")
                        else:
                            logger.error(f"❌ Failed to publish extraction result for: {job_id}")
                elif extraction_result and extraction_result.get('status') == 'FAILED':
                    # Publish failure result
                    if self.connection_manager:
                        result_success = await self.connection_manager.publish_extraction_result(extraction_result)
                        if result_success:
                            logger.info(f"⚠️ Extraction failure published for: {job_id}")
                        else:
                            logger.error(f"❌ Failed to publish extraction failure for: {job_id}")
                else:
                    logger.warning(f"⚠️ No extraction result available for: {job_id}")
            else:
                logger.error(f"❌ Extraction processing failed for: {job_id}")
            
            # Acknowledge message
            await message.ack()
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing extraction message: {str(e)}")
            # Don't requeue for code errors (like missing methods)
            # Only requeue for transient errors (network, temporary failures)
            should_requeue = not ("object has no attribute" in str(e) or "AttributeError" in str(e))
            await message.reject(requeue=should_requeue)
            return False

    async def _wait_for_extraction_completion(self, job_id: str, max_wait_time: int = 300) -> Optional[Dict[str, Any]]:
        """
        Wait for extraction to complete with polling.
        
        Args:
            job_id: The job ID to wait for
            max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
            
        Returns:
            Extraction result or None if not found
        """
        start_time = asyncio.get_event_loop().time()
        poll_interval = 2  # Check every 2 seconds
        
        while (asyncio.get_event_loop().time() - start_time) < max_wait_time:
            try:
                extraction_result = await enhanced_extraction_handler.get_extraction_result(job_id)
                
                if extraction_result:
                    status = extraction_result.get('status')
                    
                    # Normalize status to uppercase for comparison
                    status_upper = status.upper() if status else None
                    
                    if status_upper in ['COMPLETED', 'PARTIAL', 'FAILED']:
                        logger.info(f"✅ Extraction completed with status '{status}' for job: {job_id}")
                        return extraction_result
                    elif status_upper == 'PROCESSING':
                        logger.debug(f"🔄 Extraction still processing for job: {job_id}")
                    elif status_upper == 'NOT_FOUND':
                        logger.warning(f"❌ Extraction job not found: {job_id}")
                        return None
                    else:
                        logger.warning(f"⚠️ Unknown extraction status '{status}' for job: {job_id}")
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"❌ Error polling extraction result for job {job_id}: {str(e)}")
                await asyncio.sleep(poll_interval)
        
        logger.warning(f"⏰ Extraction timeout for job: {job_id} after {max_wait_time} seconds")
        return None

    def _validate_extraction_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Validate extraction message format.
        
        Args:
            message_data: Parsed message data
            
        Returns:
            bool: True if message is valid
        """
        required_fields = ["jobId", "paperId", "correlationId", "b2Url"]
        
        for field in required_fields:
            if field not in message_data:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        # Validate extraction options
        extraction_options = [
            "extractText", "extractFigures", "extractTables", "extractEquations",
            "extractCode", "extractReferences", "useOcr", "detectEntities"
        ]
        
        for option in extraction_options:
            if option not in message_data:
                logger.error(f"❌ Missing extraction option: {option}")
                return False
        
        return True

    async def _parse_message(self, message: IncomingMessage) -> Optional[Dict[str, Any]]:
        """
        Parse incoming message body.
        
        Args:
            message: Incoming RabbitMQ message
            
        Returns:
            Parsed message data or None if parsing failed
        """
        try:
            body = message.body.decode('utf-8')
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to parse message body: {str(e)}")
            return None


class MessageHandlerFactory:
    """
    Factory for creating and managing message handlers.
    
    Provides a centralized way to register and retrieve handlers
    for different message types.
    """

    def __init__(self):
        self.handlers: Dict[str, BaseMessageHandler] = {}
        self.default_handler: Optional[BaseMessageHandler] = None

    def register_handler(self, message_type: str, handler: BaseMessageHandler):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message the handler can process
            handler: Message handler instance
        """
        self.handlers[message_type] = handler
        logger.info(f"📝 Registered handler for message type: {message_type}")

    def set_default_handler(self, handler: BaseMessageHandler):
        """
        Set the default handler for unhandled message types.
        
        Args:
            handler: Default message handler
        """
        self.default_handler = handler
        logger.info("📝 Default message handler set")

    def get_handler(self, message_type: str) -> Optional[BaseMessageHandler]:
        """
        Get handler for a specific message type.
        
        Args:
            message_type: Type of message
            
        Returns:
            Message handler or default handler if not found
        """
        return self.handlers.get(message_type, self.default_handler)

    def get_all_handlers(self) -> Dict[str, BaseMessageHandler]:
        """Get all registered handlers."""
        return self.handlers.copy()
