#!/usr/bin/env python3
"""
Example script demonstrating how to use the B2 extraction API.

This script shows how to:
1. Extract content from a PDF stored in Backblaze B2
2. Handle both synchronous and asynchronous processing
3. Track job status for async requests
4. Handle errors gracefully
"""

import requests
import time
import json
from typing import Dict, Any


class B2ExtractionClient:
    """Client for the B2 extraction API"""
    
    def __init__(self, base_url: str = "http://localhost:8002/api/v1"):
        self.base_url = base_url
    
    def extract_sync(self, b2_url: str, **options) -> Dict[str, Any]:
        """
        Extract content synchronously from B2 URL
        
        Args:
            b2_url: Backblaze B2 download URL
            **options: Extraction options (extract_text, extract_figures, etc.)
        
        Returns:
            Extraction result
        """
        payload = {
            "b2_url": b2_url,
            "async_processing": False,
            **options
        }
        
        response = requests.post(f"{self.base_url}/extract-from-b2", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Extraction failed: {response.status_code} - {response.json()}")
    
    def extract_async(self, b2_url: str, **options) -> str:
        """
        Start asynchronous extraction from B2 URL
        
        Args:
            b2_url: Backblaze B2 download URL
            **options: Extraction options
        
        Returns:
            Job ID for tracking
        """
        payload = {
            "b2_url": b2_url,
            "async_processing": True,
            **options
        }
        
        response = requests.post(f"{self.base_url}/extract-from-b2", json=payload)
        
        if response.status_code == 200:
            return response.json()["job_id"]
        else:
            raise Exception(f"Failed to start extraction: {response.status_code} - {response.json()}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of extraction job"""
        response = requests.get(f"{self.base_url}/status/{job_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get job status: {response.status_code} - {response.json()}")
    
    def wait_for_completion(self, job_id: str, poll_interval: int = 5) -> Dict[str, Any]:
        """Wait for async job to complete"""
        print(f"Waiting for job {job_id} to complete...")
        
        while True:
            status = self.get_job_status(job_id)
            
            print(f"Status: {status['status']}")
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Extraction failed: {status.get('error', 'Unknown error')}")
            elif status["status"] in ["pending", "processing"]:
                time.sleep(poll_interval)
            else:
                raise Exception(f"Unknown status: {status['status']}")


def main():
    """Example usage of the B2 extraction API"""
    
    # Example B2 URL (replace with your actual B2 URL)
    b2_url = "https://f003.backblazeb2.com/b2api/v2/b2_download_file_by_id?fileId=4_z64a715e19e4932e197750a19_f1013d1b17ad2d084_d20250815_m165736_c003_v0312029_t0005_u01755277056879"
    
    client = B2ExtractionClient()
    
    # Example 1: Synchronous extraction
    print("=== Synchronous Extraction ===")
    try:
        result = client.extract_sync(
            b2_url=b2_url,
            extract_text=True,
            extract_figures=True,
            extract_tables=True,
            use_ocr=True
        )
        
        print(f"✓ Extraction completed successfully!")
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        
        if result.get('result'):
            extraction_result = result['result']
            print(f"Extracted text length: {len(extraction_result.get('text', ''))}")
            print(f"Number of figures: {len(extraction_result.get('figures', []))}")
            print(f"Number of tables: {len(extraction_result.get('tables', []))}")
            print(f"Number of sections: {len(extraction_result.get('sections', []))}")
        
    except Exception as e:
        print(f"✗ Synchronous extraction failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Asynchronous extraction
    print("=== Asynchronous Extraction ===")
    try:
        job_id = client.extract_async(
            b2_url=b2_url,
            extract_text=True,
            extract_figures=True,
            extract_tables=True,
            extract_equations=True,
            extract_references=True,
            use_ocr=True,
            detect_entities=True
        )
        
        print(f"✓ Async extraction started with job ID: {job_id}")
        
        # Wait for completion
        final_status = client.wait_for_completion(job_id)
        
        print(f"✓ Async extraction completed!")
        
        if final_status.get('result'):
            extraction_result = final_status['result']
            print(f"Extracted text length: {len(extraction_result.get('text', ''))}")
            print(f"Number of figures: {len(extraction_result.get('figures', []))}")
            print(f"Number of tables: {len(extraction_result.get('tables', []))}")
            print(f"Number of sections: {len(extraction_result.get('sections', []))}")
            
            # Save result to file
            with open(f"extraction_result_{job_id}.json", "w") as f:
                json.dump(extraction_result, f, indent=2, default=str)
            print(f"✓ Results saved to extraction_result_{job_id}.json")
        
    except Exception as e:
        print(f"✗ Asynchronous extraction failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Check service health
    print("=== Service Health Check ===")
    try:
        response = requests.get("http://localhost:8002/api/v1/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Service is healthy")
            print(f"Status: {health['status']}")
            print(f"Version: {health['version']}")
        else:
            print(f"✗ Service health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed to connect to service: {e}")


if __name__ == "__main__":
    main()
