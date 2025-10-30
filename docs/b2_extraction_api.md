# B2 Extraction API

This document describes the new API endpoint for extracting content from PDFs stored in Backblaze B2 cloud storage.

## Configuration

Before using the B2 extraction API, you need to configure the following environment variables in your `.env` file:

```env
# B2 Storage Configuration
B2_KEY_ID=your_b2_key_id
B2_APPLICATION_KEY=your_b2_application_key
B2_BUCKET_NAME=your_bucket_name
B2_BUCKET_ID=your_bucket_id  # Optional
```

## API Endpoint

### POST `/api/v1/extract-from-b2`

Extract content from a PDF stored in Backblaze B2 using a download URL.

#### Request Body

```json
{
  "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=4_z64a715e19e4932e197750a19_f1013d1b17ad2d084_d20250815_m165736_c003_v0312029_t0005_u01755277056879",
  "extract_text": true,
  "extract_figures": true,
  "extract_tables": true,
  "extract_equations": true,
  "extract_code": true,
  "extract_references": true,
  "use_ocr": true,
  "detect_entities": true,
  "async_processing": false,
  "timeout": null
}
```

#### Parameters

- `b2_url` (required): Backblaze B2 download URL with fileId parameter
- `extract_text` (optional): Extract text and structure (default: true)
- `extract_figures` (optional): Extract figures (default: true)
- `extract_tables` (optional): Extract tables (default: true)
- `extract_equations` (optional): Extract equations (default: true)
- `extract_code` (optional): Extract code blocks (default: true)
- `extract_references` (optional): Extract references (default: true)
- `use_ocr` (optional): Use OCR for scanned PDFs (default: true)
- `detect_entities` (optional): Detect named entities (default: true)
- `async_processing` (optional): Process asynchronously (default: false)
- `timeout` (optional): Extraction timeout in seconds

#### Response

**Synchronous Processing (async_processing: false):**

```json
{
  "job_id": "12345678-1234-1234-1234-123456789012",
  "status": "completed",
  "message": "B2 extraction completed successfully",
  "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=...",
  "result": {
    // Extraction result object with text, figures, tables, etc.
  }
}
```

**Asynchronous Processing (async_processing: true):**

```json
{
  "job_id": "12345678-1234-1234-1234-123456789012",
  "status": "pending",
  "message": "B2 extraction job started. Use /status endpoint to check progress.",
  "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=..."
}
```

## Status Codes

- `200 OK`: Extraction completed successfully (synchronous) or job started (asynchronous)
- `400 Bad Request`: Invalid B2 URL or downloaded content is not a valid PDF
- `404 Not Found`: File not found in B2 storage
- `500 Internal Server Error`: Extraction failed due to internal error
- `503 Service Unavailable`: B2 service not available (check credentials)

## Example Usage

### Python with requests

```python
import requests

url = "http://localhost:8002/api/v1/extract-from-b2"
payload = {
    "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=4_z64a715e19e4932e197750a19_f1013d1b17ad2d084_d20250815_m165736_c003_v0312029_t0005_u01755277056879",
    "async_processing": False
}

response = requests.post(url, json=payload)
result = response.json()

if response.status_code == 200:
    print("Extraction completed successfully!")
    print(f"Extracted {len(result['result']['text'])} characters of text")
else:
    print(f"Error: {result['message']}")
```

### cURL

```bash
curl -X POST "http://localhost:8002/api/v1/extract-from-b2" \
  -H "Content-Type: application/json" \
  -d '{
    "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=4_z64a715e19e4932e197750a19_f1013d1b17ad2d084_d20250815_m165736_c003_v0312029_t0005_u01755277056879",
    "async_processing": false
  }'
```

## Job Status Tracking

For asynchronous processing, use the existing status endpoint to track job progress:

### GET `/api/v1/status/{job_id}`

Returns job status including B2-specific information:

```json
{
  "job_id": "12345678-1234-1234-1234-123456789012",
  "status": "completed",
  "type": "b2_extraction",
  "b2_url": "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=...",
  "created_at": "2025-01-15T10:30:00Z",
  "started_at": "2025-01-15T10:30:01Z",
  "completed_at": "2025-01-15T10:30:45Z",
  "result": {
    // Extraction result object
  }
}
```

## Error Handling

The API includes comprehensive error handling:

1. **B2 URL Validation**: Checks if the URL is a valid B2 download URL
2. **B2 Service Availability**: Ensures B2 credentials are configured and service is initialized
3. **Download Verification**: Validates that downloaded content is a valid PDF
4. **Fallback Download**: If B2 SDK fails, attempts direct HTTP download
5. **Automatic Cleanup**: Temporary files are automatically cleaned up after processing

## Security Notes

- B2 credentials are stored securely in environment variables
- Temporary PDF files are cleaned up after processing
- All B2 operations use authenticated API calls
- URLs are validated to prevent abuse

## Implementation Details

The B2 extraction feature is built on top of the existing extraction pipeline and includes:

- **B2Service**: Handles B2 authentication and file downloads
- **B2ExtractionRequest/Response**: Pydantic models for API validation
- **Background Processing**: Supports both sync and async processing modes
- **Job Tracking**: Integrates with existing job management system

For more details, see the source code in:
- `app/services/b2_service.py`
- `app/models/schemas.py`
- `app/main.py` (extract-from-b2 endpoint)
