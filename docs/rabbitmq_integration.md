# RabbitMQ Integration

This document describes how the extractor service integrates with RabbitMQ for processing extraction requests from the project service.

## Architecture

The extraction system uses RabbitMQ for asynchronous communication between services:

```
Project Service -> RabbitMQ -> Extractor Service -> RabbitMQ -> Project Service
```

## Message Flow

1. **Extraction Request**: Project service sends extraction request to `scholarai.extraction.queue`
2. **Processing**: Extractor service processes the PDF extraction
3. **Completion**: Extractor service sends completion event to `scholarai.extraction.completed.queue`
4. **Database Update**: Project service updates database with extraction results

## Queue Configuration

### Exchanges
- **Exchange**: `scholarai.exchange` (topic exchange)

### Queues
- **Extraction Queue**: `scholarai.extraction.queue`
  - TTL: 10 minutes
  - Max Length: 500 messages
  - Routing Key: `scholarai.extraction`

- **Completion Queue**: `scholarai.extraction.completed.queue`
  - Routing Key: `scholarai.extraction.completed`

## Message Formats

### Extraction Request Message
```json
{
  "jobId": "uuid-string",
  "paperId": "uuid-string", 
  "correlationId": "correlation-id",
  "b2Url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=...",
  "extractText": true,
  "extractFigures": true,
  "extractTables": true,
  "extractEquations": true,
  "extractCode": true,
  "extractReferences": true,
  "useOcr": true,
  "detectEntities": true
}
```

### Completion Event Message
```json
{
  "jobId": "uuid-string",
  "paperId": "uuid-string",
  "correlationId": "correlation-id", 
  "status": "completed|failed",
  "message": "status message",
  "completedAt": "2025-01-15T10:30:00Z",
  "extractionResult": { /* full extraction JSON */ },
  "processingTime": 45.2,
  "extractionCoverage": 85.5,
  "confidenceScores": { /* confidence scores */ },
  "errors": [],
  "warnings": []
}
```

## Configuration

Add RabbitMQ configuration to `.env`:

```env
# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=scholar
RABBITMQ_PASSWORD=FindSolace@0
```

## Service Components

### RabbitMQService
- Manages RabbitMQ connections
- Declares exchanges and queues
- Handles message publishing and consumption

### ExtractionHandler  
- Processes extraction request messages
- Downloads PDFs from B2 URLs
- Runs extraction pipeline
- Sends completion messages

### BackgroundWorker
- Runs RabbitMQ consumer in background thread
- Handles graceful shutdown

## Error Handling

- **Message Validation**: Invalid messages are rejected without requeue
- **Processing Errors**: Failed extractions send failure completion events
- **Connection Issues**: Service logs errors and attempts reconnection
- **Resource Cleanup**: Temporary files are cleaned up on success or failure

## Monitoring

The service provides several endpoints for monitoring:

- `GET /api/v1/extractions` - List all extraction jobs
- `GET /api/v1/extractions/stats` - Get extraction statistics  
- `GET /api/v1/queue-status` - Check RabbitMQ connection status
- `POST /api/v1/test-b2` - Test B2 connection

## Deployment Notes

1. Ensure RabbitMQ server is running and accessible
2. Configure identical queue names in both project service and extractor
3. Set appropriate TTL and max length limits based on processing capacity
4. Monitor queue depths and processing times
5. Use persistent messages for reliable delivery
