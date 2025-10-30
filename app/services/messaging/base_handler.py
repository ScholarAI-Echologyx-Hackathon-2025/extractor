"""
Base Message Handler for ScholarAI

Provides the base interface for all message handlers in the ScholarAI
messaging system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from aio_pika import IncomingMessage

logger = logging.getLogger(__name__)


class BaseMessageHandler(ABC):
    """
    Abstract base class for all message handlers.
    
    Defines the interface that all message handlers must implement
    and provides common functionality.
    """

    def __init__(self):
        self.handler_name = self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.handler_name}")

    @abstractmethod
    async def handle_message(self, message: IncomingMessage) -> bool:
        """
        Handle an incoming message.
        
        Args:
            message: Incoming RabbitMQ message
            
        Returns:
            bool: True if message processed successfully, False otherwise
        """
        pass

    async def initialize(self, connection_manager):
        """
        Initialize the handler with connection manager.
        
        Args:
            connection_manager: RabbitMQ connection manager
        """
        self.logger.info(f"✅ {self.handler_name} initialized")

    async def cleanup(self):
        """Cleanup resources used by the handler."""
        self.logger.info(f"🔒 {self.handler_name} cleanup completed")

    def get_handler_info(self) -> Dict[str, Any]:
        """
        Get information about the handler.
        
        Returns:
            Dict containing handler information
        """
        return {
            "name": self.handler_name,
            "type": "base_handler"
        }

    async def _log_message_info(self, message: IncomingMessage):
        """Log basic message information for debugging."""
        try:
            message_size = len(message.body)
            routing_key = message.routing_key
            delivery_tag = message.delivery.delivery_tag
            
            self.logger.debug(
                f"📨 Message received - Size: {message_size} bytes, "
                f"Routing Key: {routing_key}, Delivery Tag: {delivery_tag}"
            )
        except Exception as e:
            self.logger.debug(f"Could not log message info: {str(e)}")

    async def _handle_error(self, message: IncomingMessage, error: Exception, requeue: bool = True):
        """
        Handle errors during message processing.
        
        Args:
            message: The message that caused the error
            error: The exception that occurred
            requeue: Whether to requeue the message
        """
        self.logger.error(f"❌ Error processing message: {str(error)}")
        
        try:
            if requeue:
                await message.reject(requeue=True)
                self.logger.info("🔄 Message requeued for retry")
            else:
                await message.reject(requeue=False)
                self.logger.info("🗑️ Message rejected and discarded")
        except Exception as e:
            self.logger.error(f"❌ Error handling message rejection: {str(e)}")

    def _extract_message_type(self, message: IncomingMessage) -> Optional[str]:
        """
        Extract message type from routing key or headers.
        
        Args:
            message: Incoming RabbitMQ message
            
        Returns:
            Message type string or None if not found
        """
        # Try to extract from routing key
        routing_key = message.routing_key
        if routing_key:
            parts = routing_key.split('.')
            if len(parts) > 0:
                return parts[0]
        
        # Try to extract from headers
        headers = message.header.headers
        if headers and 'message_type' in headers:
            return headers['message_type']
        
        return None
