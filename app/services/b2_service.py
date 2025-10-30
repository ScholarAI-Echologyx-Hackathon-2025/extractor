"""
Backblaze B2 Storage Service for PDF downloading and extraction.
Handles downloading PDF files from B2 cloud storage URLs.
"""

import logging
import io
from typing import Optional
from urllib.parse import urlparse, parse_qs
import httpx
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from app.config import settings

logger = logging.getLogger(__name__)


class B2Service:
    """
    Service for downloading PDF files from Backblaze B2 cloud storage.
    Provides methods for authenticating with B2 and downloading files from URLs.
    """

    def __init__(self):
        self.info = InMemoryAccountInfo()
        self.api = B2Api(self.info)
        self._authorized = False

    async def initialize(self):
        """Initialize B2 connection."""
        try:
            if not settings.b2_key_id or not settings.b2_application_key:
                raise ValueError("B2 credentials not configured")

            # Authorize the application
            self.api.authorize_account(
                "production", settings.b2_key_id, settings.b2_application_key
            )
            self._authorized = True

            logger.info("Successfully connected to B2 storage")

        except Exception as e:
            logger.error(f"Failed to initialize B2 storage: {str(e)}")
            raise

    def _ensure_authorized(self):
        """Ensure B2 is authorized before operations."""
        if not self._authorized:
            raise RuntimeError("B2 storage not initialized. Call initialize() first.")

    def _extract_file_id_from_url(self, b2_url: str) -> Optional[str]:
        """
        Extract file ID from B2 download URL.
        
        Expected URL format:
        https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=<file_id>
        """
        try:
            parsed_url = urlparse(b2_url)
            
            # Check if it's a B2 URL
            if 'backblazeb2.com' not in parsed_url.netloc:
                logger.warning(f"Not a B2 URL: {b2_url}")
                return None
            
            # Extract fileId from query parameters
            query_params = parse_qs(parsed_url.query)
            file_ids = query_params.get('fileId', [])
            
            if not file_ids:
                logger.error(f"No fileId found in B2 URL: {b2_url}")
                return None
            
            file_id = file_ids[0]
            logger.debug(f"Extracted file ID: {file_id} from URL: {b2_url}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to extract file ID from URL {b2_url}: {str(e)}")
            return None

    async def download_pdf_from_url(self, b2_url: str) -> bytes:
        """
        Download PDF content from B2 URL.
        
        Args:
            b2_url: B2 download URL with fileId parameter
            
        Returns:
            PDF content as bytes
            
        Raises:
            ValueError: If URL is invalid or file ID cannot be extracted
            RuntimeError: If B2 is not initialized or download fails
        """
        self._ensure_authorized()
        
        try:
            # Extract file ID from URL
            file_id = self._extract_file_id_from_url(b2_url)
            if not file_id:
                raise ValueError(f"Invalid B2 URL or unable to extract file ID: {b2_url}")
            
            logger.info(f"Downloading PDF from B2 with file ID: {file_id}")
            
            # Download file using B2 API
            downloaded_file = self.api.download_file_by_id(file_id)
            
            # Save to in-memory buffer to extract bytes
            buffer = io.BytesIO()
            downloaded_file.save(buffer)
            buffer.seek(0)
            pdf_content = buffer.getvalue()
            
            if not pdf_content:
                raise RuntimeError(f"Downloaded file is empty for file ID: {file_id}")
            
            # Basic PDF validation - check for PDF header
            if not pdf_content.startswith(b'%PDF'):
                raise RuntimeError(f"Downloaded content is not a valid PDF for file ID: {file_id}")
            
            logger.info(f"Successfully downloaded PDF ({len(pdf_content)} bytes) from B2")
            return pdf_content
            
        except Exception as e:
            logger.error(f"Failed to download PDF from B2 URL {b2_url}: {str(e)}")
            raise

    async def download_pdf_fallback(self, b2_url: str) -> bytes:
        """
        Fallback method to download PDF using direct HTTP request.
        This can be used if B2 SDK fails or for public URLs.
        
        Args:
            b2_url: B2 download URL
            
        Returns:
            PDF content as bytes
        """
        try:
            logger.info(f"Attempting fallback download from: {b2_url}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(b2_url)
                response.raise_for_status()
                
                pdf_content = response.content
                
                if not pdf_content:
                    raise RuntimeError("Downloaded file is empty")
                
                # Basic PDF validation
                if not pdf_content.startswith(b'%PDF'):
                    raise RuntimeError("Downloaded content is not a valid PDF")
                
                logger.info(f"Successfully downloaded PDF ({len(pdf_content)} bytes) using fallback method")
                return pdf_content
                
        except Exception as e:
            logger.error(f"Fallback download failed for {b2_url}: {str(e)}")
            raise

    def is_b2_url(self, url: str) -> bool:
        """
        Check if the provided URL is a B2 URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if it's a B2 URL, False otherwise
        """
        try:
            parsed_url = urlparse(url)
            return 'backblazeb2.com' in parsed_url.netloc and 'fileId=' in parsed_url.query
        except:
            return False




# Global service instance
b2_service = B2Service()
