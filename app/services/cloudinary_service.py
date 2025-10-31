# services/cloudinary_service.py
import asyncio
import io
from pathlib import Path
from typing import Optional, Dict, Any
import cloudinary
import cloudinary.uploader
from PIL import Image
import cv2
import numpy as np
from loguru import logger

from app.config import settings


class CloudinaryService:
    """
    Cloudinary service for uploading and managing images
    """
    
    def __init__(self):
        self.is_configured = False
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Cloudinary service"""
        try:
            if settings.cloudinary_url:
                # Parse the Cloudinary URL to extract components
                # Format: cloudinary://api_key:api_secret@cloud_name
                url = settings.cloudinary_url
                if url.startswith('cloudinary://'):
                    # Remove cloudinary:// prefix
                    url = url[13:]
                    # Split by @ to separate credentials from cloud_name
                    if '@' in url:
                        credentials, cloud_name = url.split('@', 1)
                        if ':' in credentials:
                            api_key, api_secret = credentials.split(':', 1)
                            
                            # Configure Cloudinary with individual components
                            cloudinary.config(
                                cloud_name=cloud_name,
                                api_key=api_key,
                                api_secret=api_secret
                            )
                            self.is_configured = True
                            logger.info("Cloudinary service initialized successfully")
                        else:
                            logger.error("Invalid Cloudinary URL format: missing api_secret")
                            self.is_configured = False
                    else:
                        logger.error("Invalid Cloudinary URL format: missing cloud_name")
                        self.is_configured = False
                else:
                    logger.error("Invalid Cloudinary URL format: must start with cloudinary://")
                    self.is_configured = False
            else:
                logger.warning("Cloudinary URL not configured, image uploads will be disabled")
                self.is_configured = False
        except Exception as e:
            logger.error(f"Failed to initialize Cloudinary service: {e}")
            self.is_configured = False
    
    async def upload_image_from_path(self, image_path: Path, folder: str = "scholarai/figures") -> Optional[str]:
        """Upload image from file path to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Upload image to Cloudinary
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cloudinary.uploader.upload(
                    str(image_path),
                    folder=folder,
                    resource_type='image',
                    format='png',
                    transformation=[
                        {'quality': 'auto:good'},
                        {'fetch_format': 'auto'}
                    ]
                )
            )
            
            # Return the secure URL
            cloudinary_url = result.get('secure_url')
            if cloudinary_url:
                logger.info(f"Image uploaded to Cloudinary: {cloudinary_url}")
                return cloudinary_url
            else:
                logger.error("Failed to get Cloudinary URL from upload result")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload image to Cloudinary: {e}")
            return None
    
    async def upload_image_from_bytes(self, image_bytes: bytes, filename: str, folder: str = "scholarai/figures") -> Optional[str]:
        """Upload image from bytes to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Upload image bytes to Cloudinary
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cloudinary.uploader.upload(
                    io.BytesIO(image_bytes),
                    public_id=filename,
                    folder=folder,
                    resource_type='image',
                    format='png',
                    transformation=[
                        {'quality': 'auto:good'},
                        {'fetch_format': 'auto'}
                    ]
                )
            )
            
            # Return the secure URL
            cloudinary_url = result.get('secure_url')
            if cloudinary_url:
                logger.info(f"Image uploaded to Cloudinary: {cloudinary_url}")
                return cloudinary_url
            else:
                logger.error("Failed to get Cloudinary URL from upload result")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload image to Cloudinary: {e}")
            return None
    
    async def upload_cv_image(self, cv_image: np.ndarray, filename: str, folder: str = "scholarai/figures") -> Optional[str]:
        """Upload OpenCV image to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Convert OpenCV image to bytes
            success, buffer = cv2.imencode('.png', cv_image)
            if not success:
                logger.error("Failed to encode OpenCV image")
                return None
            
            image_bytes = buffer.tobytes()
            return await self.upload_image_from_bytes(image_bytes, filename, folder)
            
        except Exception as e:
            logger.error(f"Failed to upload OpenCV image to Cloudinary: {e}")
            return None
    
    async def upload_pil_image(self, pil_image: Image.Image, filename: str, folder: str = "scholarai/figures") -> Optional[str]:
        """Upload PIL image to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return await self.upload_image_from_bytes(img_byte_arr, filename, folder)
            
        except Exception as e:
            logger.error(f"Failed to upload PIL image to Cloudinary: {e}")
            return None
    
    async def upload_file(self, file_path: str, folder: str = "scholarai/files", public_id: str = None) -> Optional[str]:
        """Upload any file to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Determine resource type based on file extension
            file_extension = Path(file_path).suffix.lower()
            resource_type = 'auto'  # default
            
            # For CSV files, use 'raw' resource type
            if file_extension == '.csv':
                resource_type = 'raw'
            elif file_extension in ['.pdf', '.doc', '.docx', '.txt']:
                resource_type = 'raw'
            
            # Upload file to Cloudinary
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cloudinary.uploader.upload(
                    file_path,
                    folder=folder,
                    public_id=public_id,
                    resource_type=resource_type
                )
            )
            
            # Return the secure URL
            cloudinary_url = result.get('secure_url')
            if cloudinary_url:
                logger.info(f"File uploaded to Cloudinary: {cloudinary_url}")
                return cloudinary_url
            else:
                logger.error("Failed to get Cloudinary URL from upload result")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload file to Cloudinary: {e}")
            return None
    
    async def upload_bytes(self, file_bytes: bytes, folder: str = "scholarai/files", public_id: str = None, file_extension: str = ".txt") -> Optional[str]:
        """Upload bytes to Cloudinary"""
        if not self.is_configured:
            logger.warning("Cloudinary not configured, skipping upload")
            return None
        
        try:
            # Determine resource type based on file extension
            resource_type = 'raw'  # default for bytes
            
            # For CSV files, use 'raw' resource type
            if file_extension == '.csv':
                resource_type = 'raw'
            elif file_extension in ['.pdf', '.doc', '.docx', '.txt']:
                resource_type = 'raw'
            
            # Create a temporary file-like object
            file_obj = io.BytesIO(file_bytes)
            
            # Upload bytes to Cloudinary
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: cloudinary.uploader.upload(
                    file_obj,
                    folder=folder,
                    public_id=public_id,
                    resource_type=resource_type
                )
            )
            
            # Return the secure URL
            cloudinary_url = result.get('secure_url')
            if cloudinary_url:
                logger.info(f"Bytes uploaded to Cloudinary: {cloudinary_url}")
                return cloudinary_url
            else:
                logger.error("Failed to get Cloudinary URL from upload result")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload bytes to Cloudinary: {e}")
            return None
    
    def is_service_available(self) -> bool:
        """Check if Cloudinary service is available"""
        return self.is_configured


# Create global instance
cloudinary_service = CloudinaryService()
