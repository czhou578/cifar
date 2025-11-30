#!/usr/bin/env python3
"""
Stress test script for CIFAR-100 API /predict endpoint
Makes 50 concurrent requests to test API reliability under load
"""

import asyncio
import aiohttp
import time
from pathlib import Path
import io
from PIL import Image
import numpy as np
import statistics
import sys

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
NUM_CONCURRENT_REQUESTS = 100
TOP_K = 5

def create_test_image():
    """Create a synthetic CIFAR-100 style test image (32x32 RGB)"""
    # Create a colorful test pattern that resembles a real image
    image_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Add some structure to make it look more like a real image
    # Create a simple pattern with gradients
    for i in range(32):
        for j in range(32):
            # Create a radial gradient pattern
            center_x, center_y = 16, 16
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            
            # Modify colors based on distance from center
            image_array[i, j, 0] = min(255, int(128 + 50 * np.sin(distance * 0.5)))  # Red
            image_array[i, j, 1] = min(255, int(128 + 50 * np.cos(distance * 0.3)))  # Green  
            image_array[i, j, 2] = min(255, int(128 + 50 * np.sin(distance * 0.7)))  # Blue
    
    # Convert to PIL Image
    image = Image.fromarray(image_array, 'RGB')
    
    # Convert to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)
    
    return img_byte_array.getvalue()

async def make_prediction_request(session: aiohttp.ClientSession, request_id: int, image_bytes: bytes):
    """Make a single prediction request"""
    start_time = time.time()
    
    try:
        # Prepare the multipart form data
        data = aiohttp.FormData()
        data.add_field('file', 
                      image_bytes, 
                      filename=f'test_image_{request_id}.png',
                      content_type='image/png')
        data.add_field('top_k', str(TOP_K))
        
        async with session.post(PREDICT_ENDPOINT, data=data) as response:
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status == 200:
                result = await response.json()
                return {
                    'request_id': request_id,
                    'status': 'success',
                    'response_time': response_time,
                    'predictions': result.get('predictions', []),
                    'cached': result.get('cached', False),
                    'filename': result.get('filename', ''),
                    'error': None
                }
            else:
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'status': 'error',
                    'response_time': response_time,
                    'predictions': [],
                    'cached': False,
                    'filename': '',
                    'error': f"HTTP {response.status}: {error_text}"
                }
                
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return {
            'request_id': request_id,
            'status': 'error',
            'response_time': response_time,
            'predictions': [],
            'cached': False,
            'filename': '',
            'error': str(e)
        }

async def run_stress_test():
    """Run the stress test with concurrent requests"""
    print(f"🚀 Starting stress test with {NUM_CONCURRENT_REQUESTS} concurrent requests")
    print(f"📡 Target endpoint: {PREDICT_ENDPOINT}")
    print(f"🎯 Top-K predictions: {TOP_K}")
    print("-" * 60)
    
    # Create test image
    print("📸 Creating test image...")
    image_bytes = create_test_image()
    print(f"✅ Test image created ({len(image_bytes)} bytes)")
    
    # Setup HTTP session with appropriate timeouts
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=60)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Create all request tasks
        tasks = [
            make_prediction_request(session, i, image_bytes) 
            for i in range(NUM_CONCURRENT_REQUESTS)
        ]
        
        print(f"⏱️  Sending {NUM_CONCURRENT_REQUESTS} concurrent requests...")
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Process results
    successful_requests = []
    failed_requests = []
    cached_requests = []
    
    for result in results:
        if isinstance(result, Exception):
            failed_requests.append({
                'request_id': 'unknown',
                'status': 'exception',
                'error': str(result),
                'response_time': 0
            })
        elif result['status'] == 'success':
            successful_requests.append(result)
            if result['cached']:
                cached_requests.append(result)
        else:
            failed_requests.append(result)
    
    # Calculate statistics
    if successful_requests:
        response_times = [r['response_time'] for r in successful_requests]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        median_response_time = statistics.median(response_times)
        
        if len(response_times) > 1:
            std_response_time = statistics.stdev(response_times)
        else:
            std_response_time = 0
    else:
        avg_response_time = 0
        min_response_time = 0
        max_response_time = 0
        median_response_time = 0
        std_response_time = 0
    
    # Print results
    print("\n" + "="*60)
    print("📊 STRESS TEST RESULTS")
    print("="*60)
    print(f"⏱️  Total Test Duration: {total_time:.2f} seconds")
    print(f"📈 Requests per Second: {NUM_CONCURRENT_REQUESTS/total_time:.2f}")
    print()
    print(f"✅ Successful Requests: {len(successful_requests)}/{NUM_CONCURRENT_REQUESTS} ({len(successful_requests)/NUM_CONCURRENT_REQUESTS*100:.1f}%)")
    print(f"❌ Failed Requests: {len(failed_requests)}/{NUM_CONCURRENT_REQUESTS} ({len(failed_requests)/NUM_CONCURRENT_REQUESTS*100:.1f}%)")
    print(f"💾 Cached Responses: {len(cached_requests)}/{len(successful_requests)} ({len(cached_requests)/max(1,len(successful_requests))*100:.1f}%)")
    print()
    print("⏱️  Response Time Statistics:")
    print(f"   • Average: {avg_response_time:.3f}s")
    print(f"   • Median:  {median_response_time:.3f}s")
    print(f"   • Min:     {min_response_time:.3f}s")
    print(f"   • Max:     {max_response_time:.3f}s")
    print(f"   • Std Dev: {std_response_time:.3f}s")
    
    # Show sample predictions from successful requests
    if successful_requests:
        print("\n🎯 Sample Predictions (from first successful request):")
        sample = successful_requests[0]
        print(f"   File: {sample['filename']}")
        for i, pred in enumerate(sample['predictions'][:3]):  # Show top 3
            print(f"   {i+1}. {pred['class_name']}: {pred['confidence']:.1%}")
    
    # Show errors if any
    if failed_requests:
        print(f"\n❌ Error Summary ({len(failed_requests)} failures):")
        error_counts = {}
        for req in failed_requests:
            error = req['error'] or 'Unknown error'
            error_counts[error] = error_counts.get(error, 0) + 1
        
        for error, count in error_counts.items():
            print(f"   • {error}: {count} occurrences")
    
    print("\n" + "="*60)
    
    # Return summary for programmatic use
    return {
        'total_requests': NUM_CONCURRENT_REQUESTS,
        'successful_requests': len(successful_requests),
        'failed_requests': len(failed_requests),
        'cached_requests': len(cached_requests),
        'total_time': total_time,
        'avg_response_time': avg_response_time,
        'requests_per_second': NUM_CONCURRENT_REQUESTS/total_time
    }

async def health_check():
    """Check if the API is running before starting the stress test"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:8000/health") as response:
                if response.status == 200:
                    return True
                else:
                    print(f"❌ Health check failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   Make sure your FastAPI server is running on {API_BASE_URL}")
        return False

if __name__ == "__main__":
    print("🧪 CIFAR-100 API Stress Test")
    print("="*60)
    
    # Check if API is available
    print("🔍 Checking API health...")
    if not asyncio.run(health_check()):
        print("❌ API health check failed. Please start your FastAPI server first.")
        sys.exit(1)
    
    print("✅ API is responding")
    print()
    
    # Run the stress test
    try:
        summary = asyncio.run(run_stress_test())
        
        # Exit with appropriate code
        if summary['failed_requests'] == 0:
            print("🎉 All requests successful!")
            sys.exit(0)
        elif summary['successful_requests'] > summary['failed_requests']:
            print("⚠️  Some requests failed, but majority succeeded")
            sys.exit(0)
        else:
            print("💥 Most requests failed - check your API")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)
