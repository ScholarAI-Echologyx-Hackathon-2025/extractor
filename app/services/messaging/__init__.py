"""
Messaging module for ScholarAI Extractor

Provides RabbitMQ connection management, message handling, and consumer
functionality for extraction tasks.
"""

from .connection import RabbitMQConnection
from .consumer import ScholarAIConsumer
from .handlers import ExtractionMessageHandler, MessageHandlerFactory
from .base_handler import BaseMessageHandler

__all__ = [
    "RabbitMQConnection",
    "ScholarAIConsumer", 
    "ExtractionMessageHandler",
    "MessageHandlerFactory",
    "BaseMessageHandler"
]
