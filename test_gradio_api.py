#!/usr/bin/env python3
"""
Test Gradio API Connection
Test if we can call the Gradio model directly
"""

import requests
import json

def test_gradio_api(port=7861):
    """Test the Gradio API"""
    print(f"🧪 Testing Gradio API on port {port}")
    
    # Test basic connection
    try:
        response = requests.get(f"http://localhost:{port}", timeout=5)
        print(f"✅ Gradio server is running (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Cannot reach Gradio server: {e}")
        return False
    
    # Test API endpoint (try both real Gradio and mock server)
    try:
        # First try real Gradio API
        api_url = f"http://localhost:{port}/api/predict"
        payload = {
            "data": [
                "What is organic farming?",  # prompt
                256,                         # max_tokens
                0.7,                        # temperature
                0.9                         # top_p
            ]
        }
        
        print(f"📡 Trying Gradio API: {api_url}")
        response = requests.post(api_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Gradio API Response: {json.dumps(result, indent=2)}")
            return True
        elif response.status_code == 404:
            print(f"⚠️ Gradio API not found, trying mock server API...")
            # Try mock server API
            return test_mock_api(port)
        else:
            print(f"❌ Gradio API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"⚠️ Gradio API test failed: {e}, trying mock server...")
        return test_mock_api(port)

def test_mock_api(port):
    """Test the mock server API"""
    try:
        api_url = f"http://localhost:{port}/api/predict"
        payload = {
            "data": [
                "What is organic farming?",  # prompt
                256,                         # max_tokens
                0.7,                        # temperature
                0.9                         # top_p
            ]
        }
        
        print(f"📡 Calling Mock API: {api_url}")
        print(f"📝 Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(api_url, json=payload, timeout=10)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Mock API Response: {json.dumps(result, indent=2)}")
            
            if "data" in result and len(result["data"]) > 0:
                model_response = result["data"][0]
                print(f"🤖 Mock Model Response: {model_response}")
                return True
            else:
                print("⚠️ Unexpected response format")
                return False
        else:
            print(f"❌ Mock API call failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Mock API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gradio_api()
    if success:
        print("🎉 Gradio API is working!")
    else:
        print("❌ Gradio API test failed")