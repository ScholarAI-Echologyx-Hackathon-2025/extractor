"""
RabbitMQ service module.

This module demonstrates how to use the central configuration system
in service classes throughout the application.
"""

import logging
from typing import Optional
from app.config import settings


class RabbitMQService:
    """
    RabbitMQ service for handling message queue operations.
    
    This class demonstrates how to use the central configuration
    system to access environment variables in service classes.
    """
    
    def __init__(self):
        """Initialize RabbitMQ service with configuration from settings."""
        self.logger = logging.getLogger(__name__)
        self.host = settings.rabbitmq_host
        self.port = settings.rabbitmq_port
        self.username = settings.rabbitmq_user
        self.password = settings.rabbitmq_password
        self._connection: Optional[object] = None
    
    async def connect(self) -> bool:
        """
        Connect to RabbitMQ server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to RabbitMQ at {self.host}:{self.port}")
            
            # Here you would implement actual RabbitMQ connection logic
            # For demonstration, we'll just simulate a connection
            
            self.logger.info("Successfully connected to RabbitMQ")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ server."""
        if self._connection:
            self.logger.info("Disconnecting from RabbitMQ")
            # Implement actual disconnection logic here
            self._connection = None
    
    async def publish_message(self, queue: str, message: dict) -> bool:
        """
        Publish a message to a queue.
        
        Args:
            queue: Queue name
            message: Message to publish
            
        Returns:
            bool: True if message published successfully
        """
        try:
            self.logger.debug(f"Publishing message to queue '{queue}': {message}")
            
            # Here you would implement actual message publishing logic
            # using the configuration from settings
            
            self.logger.info(f"Message published to queue '{queue}' successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message to queue '{queue}': {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        Get connection information for debugging.
        
        Returns:
            dict: Connection information (without sensitive data)
        """
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "debug": settings.debug,
            "log_level": settings.log_level
        }


# Example of how to create a singleton service instance
_rabbitmq_service: Optional[RabbitMQService] = None


def get_rabbitmq_service() -> RabbitMQService:
    """
    Get RabbitMQ service instance (singleton pattern).
    
    Returns:
        RabbitMQService: The RabbitMQ service instance
    """
    global _rabbitmq_service
    if _rabbitmq_service is None:
        _rabbitmq_service = RabbitMQService()
    return _rabbitmq_service


# Global instance for backward compatibility
rabbitmq_service = get_rabbitmq_service() 