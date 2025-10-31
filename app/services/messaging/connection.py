"""
RabbitMQ Connection Manager for ScholarAI Extractor

Handles connection management, queue setup, and message routing for
the ScholarAI extraction messaging system.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from aio_pika import connect_robust, Message, ExchangeType, Queue, Connection, Channel
from app.config import settings

logger = logging.getLogger(__name__)


class RabbitMQConnection:
    """
    Manages RabbitMQ connections, exchanges, and queues for ScholarAI Extractor.
    
    Handles connection lifecycle, queue setup, and provides interfaces
    for message publishing and consumption.
    """

    def __init__(self):
        # Connection state
        self.connection: Optional[Connection] = None
        self.channel: Optional[Channel] = None
        
        # Queues and exchanges
        self.extraction_queue: Optional[Queue] = None
        self.extraction_exchange = None
        
        # Connection status
        self.is_connected = False
        self.is_setup = False

    async def connect(self) -> bool:
        """
        Establish connection to RabbitMQ server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"🔗 Connecting to RabbitMQ at {settings.rabbitmq_host}:{settings.rabbitmq_port}")
            
            # Build connection URL
            if settings.rabbitmq_user and settings.rabbitmq_password:
                connection_url = (
                    f"amqp://{settings.rabbitmq_user}:{settings.rabbitmq_password}"
                    f"@{settings.rabbitmq_host}:{settings.rabbitmq_port}/"
                )
            else:
                connection_url = f"amqp://{settings.rabbitmq_host}:{settings.rabbitmq_port}/"
            
            # Establish connection
            self.connection = await connect_robust(connection_url)
            self.channel = await self.connection.channel()
            
            self.is_connected = True
            logger.info("✅ RabbitMQ connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {str(e)}")
            self.is_connected = False
            return False

    async def setup_queues(self) -> bool:
        """
        Setup exchanges and queues for ScholarAI extraction messaging.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        if not self.is_connected or not self.channel:
            logger.error("❌ Cannot setup queues: not connected to RabbitMQ")
            return False
            
        try:
            logger.info("🛠️ Setting up RabbitMQ exchanges and queues for extraction...")
            
            # Declare exchange
            self.extraction_exchange = await self.channel.declare_exchange(
                settings.rabbitmq_exchange,
                ExchangeType.TOPIC,
                durable=True
            )
            
            # Declare extraction queue
            self.extraction_queue = await self.channel.declare_queue(
                settings.rabbitmq_extraction_queue,
                durable=True,
                arguments={
                    "x-message-ttl": 600000,  # 10 minutes TTL for longer extraction tasks
                    "x-max-length": 500,      # Max 500 messages
                }
            )
            
            # Bind queue to exchange
            await self.extraction_queue.bind(
                self.extraction_exchange,
                routing_key=settings.rabbitmq_extraction_routing_key
            )
            
            self.is_setup = True
            logger.info("✅ RabbitMQ extraction queues and exchanges setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup extraction queues: {str(e)}")
            self.is_setup = False
            return False

    async def publish_message(
        self, 
        routing_key: str, 
        message_data: Dict[str, Any],
        exchange_name: Optional[str] = None
    ) -> bool:
        """
        Publish a message to RabbitMQ.
        
        Args:
            routing_key: Message routing key
            message_data: Message data to publish
            exchange_name: Exchange name (uses default if not specified)
            
        Returns:
            bool: True if message published successfully
        """
        if not self.is_connected or not self.channel:
            logger.error("❌ Cannot publish message: not connected to RabbitMQ")
            return False
            
        try:
            import json
            
            # Serialize message
            message_body = json.dumps(message_data).encode()
            
            # Create message
            message = Message(
                body=message_body,
                delivery_mode=2,  # Persistent
                content_type="application/json"
            )
            
            # Get exchange
            exchange = self.extraction_exchange
            if exchange_name:
                exchange = await self.channel.get_exchange(exchange_name)
            
            # Publish message
            await exchange.publish(message, routing_key=routing_key)
            
            logger.debug(f"📤 Message published to {routing_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {str(e)}")
            return False

    async def publish_extraction_result(self, result: Dict[str, Any]) -> bool:
        """
        Publish extraction result to the result queue.
        
        Args:
            result: Extraction result data
            
        Returns:
            bool: True if result published successfully
        """
        return await self.publish_message(
            routing_key=settings.rabbitmq_extraction_completed_routing_key,
            message_data=result
        )

    def get_extraction_queue(self) -> Optional[Queue]:
        """Get the extraction queue instance."""
        return self.extraction_queue

    def get_websearch_queue(self) -> Optional[Queue]:
        """Get the websearch queue instance (for compatibility)."""
        return None  # Not implemented in extractor

    def get_structuring_queue(self) -> Optional[Queue]:
        """Get the structuring queue instance (for compatibility)."""
        return None  # Not implemented in extractor

    async def close(self):
        """Close RabbitMQ connection and cleanup resources."""
        try:
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
                
            self.is_connected = False
            self.is_setup = False
            logger.info("🔒 RabbitMQ connection closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing RabbitMQ connection: {str(e)}")

    def is_healthy(self) -> bool:
        """Check if the connection is healthy."""
        return self.is_connected and self.is_setup
