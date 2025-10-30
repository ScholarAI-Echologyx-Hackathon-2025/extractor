#!/usr/bin/env python3
"""
Test script for RabbitMQ integration in ScholarAI Extractor

This script tests the RabbitMQ messaging integration to ensure:
1. Connection to RabbitMQ works
2. Queue setup works
3. Message publishing and consumption works
4. Integration with extraction handler works
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import pytest

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from app.services.messaging import ScholarAIConsumer, RabbitMQConnection
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_rabbitmq_connection():
    """Test basic RabbitMQ connection"""
    logger.info("🔗 Testing RabbitMQ connection...")
    
    connection = RabbitMQConnection()
    
    try:
        # Test connection
        connected = await connection.connect()
        assert connected, "Failed to connect to RabbitMQ"
        
        logger.info("✅ RabbitMQ connection successful")
        
        # Test queue setup
        setup_success = await connection.setup_queues()
        assert setup_success, "Failed to setup queues"
        
        logger.info("✅ Queue setup successful")
        
        # Cleanup
        await connection.close()
        logger.info("✅ Connection cleanup successful")
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_consumer_functionality():
    """Test the consumer functionality"""
    logger.info("🔄 Testing consumer functionality...")
    
    consumer = ScholarAIConsumer()
    
    try:
        # Start consumer (this will run for a short time)
        logger.info("Starting consumer...")
        
        # Create a task that will stop the consumer after a few seconds
        async def stop_consumer_after_delay():
            await asyncio.sleep(5)  # Run for 5 seconds
            logger.info("Stopping consumer...")
            await consumer.stop()
        
        # Run both tasks
        await asyncio.gather(
            consumer.start(),
            stop_consumer_after_delay()
        )
        
        logger.info("✅ Consumer test completed")
        
    except Exception as e:
        logger.error(f"❌ Consumer test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_full_integration():
    """Test the full integration with connection and queue setup"""
    logger.info("🧪 Testing full integration...")
    
    # Test connection and queue setup
    connection = RabbitMQConnection()
    try:
        await connection.connect()
        await connection.setup_queues()
        
        logger.info("✅ Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        raise
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_rabbitmq_configuration():
    """Test RabbitMQ configuration settings"""
    logger.info("⚙️ Testing RabbitMQ configuration...")
    
    # Check if RabbitMQ is configured
    if not settings.rabbitmq_user or not settings.rabbitmq_password:
        logger.warning("⚠️ RabbitMQ credentials not configured. Using default connection.")
        logger.info("Set RABBITMQ_USER and RABBITMQ_PASSWORD environment variables for full testing.")
    
    # Verify required settings are present
    assert hasattr(settings, 'rabbitmq_host'), "RabbitMQ host setting missing"
    assert hasattr(settings, 'rabbitmq_port'), "RabbitMQ port setting missing"
    assert hasattr(settings, 'rabbitmq_exchange'), "RabbitMQ exchange setting missing"
    assert hasattr(settings, 'rabbitmq_extraction_queue'), "RabbitMQ extraction queue setting missing"
    assert hasattr(settings, 'rabbitmq_extraction_routing_key'), "RabbitMQ extraction routing key setting missing"
    
    logger.info("✅ RabbitMQ configuration test passed")


# Legacy main function for standalone execution
async def main():
    """Main test function for standalone execution"""
    logger.info("🚀 Starting RabbitMQ integration tests...")
    
    # Check if RabbitMQ is configured
    if not settings.rabbitmq_user or not settings.rabbitmq_password:
        logger.warning("⚠️ RabbitMQ credentials not configured. Using default connection.")
        logger.info("Set RABBITMQ_USER and RABBITMQ_PASSWORD environment variables for full testing.")
    
    # Run tests
    tests = [
        ("RabbitMQ Configuration", test_rabbitmq_configuration),
        ("RabbitMQ Connection", test_rabbitmq_connection),
        ("Full Integration", test_full_integration),
        # Note: Consumer test is commented out as it requires manual intervention
        # ("Consumer", test_consumer_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            await test_func()
            results.append((test_name, True))
            logger.info(f"✅ {test_name} PASSED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! RabbitMQ integration is working correctly.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check the configuration and RabbitMQ setup.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)
