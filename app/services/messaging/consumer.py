"""
Main RabbitMQ consumer for ScholarAI Extractor

Orchestrates connection management, message routing, and result publishing
with a clean, modular architecture for extraction tasks.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from aio_pika import IncomingMessage

from .connection import RabbitMQConnection
from .handlers import ExtractionMessageHandler, MessageHandlerFactory

logger = logging.getLogger(__name__)


class ScholarAIConsumer:
    """
    Main RabbitMQ consumer for ScholarAI Extractor services.

    Orchestrates connection management, message routing, and result publishing
    with a clean, modular architecture for extraction tasks.
    """

    def __init__(self):
        # Initialize connection manager
        self.connection_manager = RabbitMQConnection()

        # Initialize message handler factory
        self.handler_factory = MessageHandlerFactory()
        self._setup_handlers()

        # State tracking
        self.is_running = False

    def _setup_handlers(self):
        """Setup message handlers for different message types"""
        # Register extraction handler
        extraction_handler = ExtractionMessageHandler()
        self.handler_factory.register_handler("scholarai", extraction_handler)
        self.handler_factory.set_default_handler(extraction_handler)

        logger.info("📝 Extraction message handlers ready")

    async def start(self):
        """
        Start the consumer and begin processing messages.
        """
        try:
            logger.info("🚀 Starting ScholarAI Extractor Consumer...")

            # Connect to RabbitMQ
            connected = await self.connection_manager.connect()
            if not connected:
                raise RuntimeError("Failed to connect to RabbitMQ")

            # Setup queues and exchanges
            setup_success = await self.connection_manager.setup_queues()
            if not setup_success:
                raise RuntimeError("Failed to setup RabbitMQ queues")

            # Start consuming messages
            await self._start_consuming()

        except Exception as e:
            logger.error(f"❌ Failed to start consumer: {e}")
            await self.stop()
            raise

    async def _start_consuming(self):
        """Start consuming messages from extraction queue"""
        extraction_queue = self.connection_manager.get_extraction_queue()

        if not extraction_queue:
            raise RuntimeError("Extraction queue not available")

        # Initialize handlers with connection manager
        for handler in self.handler_factory.get_all_handlers().values():
            await handler.initialize(self.connection_manager)

        logger.info("🔄 Starting extraction message consumption...")
        self.is_running = True

        # Start consuming messages
        async with extraction_queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self.is_running:
                    break
                
                await self._process_message(message)

    async def _process_message(self, message: IncomingMessage):
        """
        Process a single message using appropriate handler.
        
        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Extract message type
            message_type = self._extract_message_type(message)
            
            # Get appropriate handler
            handler = self.handler_factory.get_handler(message_type)
            if not handler:
                logger.warning(f"⚠️ No handler found for message type: {message_type}")
                await message.reject(requeue=False)
                return
            
            # Process message
            success = await handler.handle_message(message)
            
            if not success:
                logger.error(f"❌ Message processing failed for type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {str(e)}")
            await message.reject(requeue=True)

    def _extract_message_type(self, message: IncomingMessage) -> str:
        """
        Extract message type from routing key.
        
        Args:
            message: Incoming RabbitMQ message
            
        Returns:
            Message type string
        """
        routing_key = message.routing_key
        if routing_key:
            parts = routing_key.split('.')
            if len(parts) > 0:
                return parts[0]
        
        return "extraction"  # Default to extraction

    async def stop(self):
        """
        Stop the consumer and cleanup resources.
        """
        logger.info("🛑 Stopping ScholarAI Extractor Consumer...")
        self.is_running = False

        # Cleanup handlers
        for handler in self.handler_factory.get_all_handlers().values():
            await handler.cleanup()

        # Close connection
        await self.connection_manager.close()
        
        logger.info("✅ ScholarAI Extractor Consumer stopped")

    def get_status(self) -> Dict[str, Any]:
        """
        Get consumer status information.
        
        Returns:
            Dict containing status information
        """
        return {
            "is_running": self.is_running,
            "connection_healthy": self.connection_manager.is_healthy(),
            "handlers": {
                name: handler.get_handler_info() 
                for name, handler in self.handler_factory.get_all_handlers().items()
            }
        }
