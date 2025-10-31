"""
Enhanced background worker for RabbitMQ message consumption with adaptive processing.
"""

import asyncio
import logging
import threading
from app.services.messaging import ScholarAIConsumer
from app.services.extraction_handler import enhanced_extraction_handler

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """Enhanced background worker to run RabbitMQ consumer with adaptive processing."""

    def __init__(self):
        self.worker_thread = None
        self.running = False
        self.enhanced_handler = enhanced_extraction_handler
        self.consumer = ScholarAIConsumer()

    def start(self):
        """Start the background worker."""
        if self.running:
            logger.warning("Background worker is already running")
            return

        logger.info("Starting enhanced background worker for RabbitMQ")
        self.running = True
        self.worker_thread = threading.Thread(target=self._run_consumer, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the background worker."""
        if not self.running:
            return

        logger.info("Stopping enhanced background worker")
        self.running = False
        
        try:
            # Run async stop in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.consumer.stop())
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error stopping RabbitMQ consumer: {e}")

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def _run_consumer(self):
        """Run the enhanced RabbitMQ consumer with adaptive processing."""
        try:
            logger.info("Starting enhanced RabbitMQ message consumption with adaptive processing")
            
            # Run async start in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Start the consumer which handles connection and message processing
                loop.run_until_complete(self.consumer.start())
            except Exception as e:
                logger.error(f"Error in RabbitMQ consumer: {e}")
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Error in enhanced RabbitMQ consumer: {e}")
            self.running = False

    async def get_performance_summary(self):
        """Get performance summary from enhanced extraction handler."""
        try:
            return await self.enhanced_handler.get_performance_summary()
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}


# Global worker instance
background_worker = BackgroundWorker()
