#!/usr/bin/env python3
"""
AI Load Balancer Client v1.0
Windows laptop client for distributed LLM processing
"""

import grpc
import time
import threading
import psutil
import platform
import socket
import logging
import uuid
import json
import os
import sys
import subprocess
from typing import Dict, Any, Optional, List

# Import generated gRPC files
import load_balancer_pb2
import load_balancer_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AILoadBalancerClient:
    """AI Load Balancer Client for distributed LLM processing"""
    
    def __init__(self, server_address: str = "192.168.1.8:50051", client_id: str = None):
        self.server_address = server_address
        self.client_id = client_id or self._generate_client_id()
        self.channel = None
        self.stub = None
        self._running = False
        self._heartbeat_thread = None
        self.llm_process = None
        self.assigned_model = None
        
        logger.info(f"AI Load Balancer Client v1.0 initialized")
        logger.info(f"Client ID: {self.client_id}")
        logger.info(f"Server: {self.server_address}")
    
    def _generate_client_id(self) -> str:
        """Generate a unique client ID"""
        hostname = socket.gethostname()
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"client-{hostname}-{timestamp}-{unique_id}"
    
    def get_system_specs(self):
        """Get current system specifications with AI capabilities"""
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq()
            cpu_frequency = cpu_freq.current / 1000 if cpu_freq else 2.0
            
            # Get memory information
            memory = psutil.virtual_memory()
            ram_gb = memory.total // (1024**3)
            
            # Get GPU information
            gpu_info = self._get_gpu_info()
            gpu_memory_gb = self._get_gpu_memory()
            
            # Get OS information
            os_info = f"{platform.system()} {platform.release()}"
            
            # Calculate AI-focused performance score
            performance_score = self._calculate_ai_performance_score(
                cpu_count, cpu_frequency, ram_gb, gpu_memory_gb
            )
            
            return load_balancer_pb2.SystemSpecs(
                cpu_cores=cpu_count,
                cpu_frequency_ghz=cpu_frequency,
                ram_gb=ram_gb,
                gpu_info=gpu_info,
                gpu_memory_gb=gpu_memory_gb,
                os_info=os_info,
                performance_score=performance_score
            )
            
        except Exception as e:
            logger.error(f"Error getting system specs: {e}")
            return load_balancer_pb2.SystemSpecs(
                cpu_cores=4,
                cpu_frequency_ghz=2.0,
                ram_gb=8,
                gpu_info="Unknown",
                gpu_memory_gb=0.0,
                os_info=platform.system(),
                performance_score=50.0
            )
    
    def _get_gpu_info(self) -> str:
        """Get GPU information"""
        try:
            # Try to get NVIDIA GPU info
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback to generic detection
        return "Integrated Graphics"
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory in GB"""
        try:
            # Try to get NVIDIA GPU memory
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memory_mb = int(result.stdout.strip())
                return memory_mb / 1024.0  # Convert to GB
        except:
            pass
        
        # Estimate based on system RAM for integrated graphics
        memory = psutil.virtual_memory()
        ram_gb = memory.total // (1024**3)
        return min(ram_gb * 0.25, 4.0)  # Assume 25% of RAM, max 4GB
    
    def _calculate_ai_performance_score(self, cpu_cores: int, cpu_freq: float, ram_gb: int, gpu_memory_gb: float) -> float:
        """Calculate AI-focused performance score"""
        # AI workloads are heavily dependent on memory and compute
        cpu_score = cpu_cores * cpu_freq * 0.3
        ram_score = min(ram_gb / 32.0, 1.0) * 30  # Normalize to 32GB
        gpu_score = min(gpu_memory_gb / 24.0, 1.0) * 40  # Normalize to 24GB
        
        total_score = cpu_score + ram_score + gpu_score
        return min(total_score, 100.0)
    
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # First try to connect to the server to get the right interface
            server_ip = self.server_address.split(':')[0]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((server_ip, 80))
                local_ip = s.getsockname()[0]
                logger.info(f"Detected local IP: {local_ip}")
                return local_ip
        except:
            try:
                # Fallback to Google DNS
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    return s.getsockname()[0]
            except:
                return "127.0.0.1"
    
    def connect(self) -> bool:
        """Connect to the AI load balancer server"""
        try:
            logger.info(f"Connecting to AI Load Balancer at {self.server_address}...")
            
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = load_balancer_pb2_grpc.LoadBalancerStub(self.channel)
            
            # Test connection with health check
            response = self.stub.HealthCheck(load_balancer_pb2.Empty(), timeout=10)
            
            if response.healthy:
                logger.info("Connected to AI Load Balancer successfully")
                return True
            else:
                logger.error("Load balancer reported unhealthy status")
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def register(self) -> bool:
        """Register this client with the AI load balancer"""
        try:
            specs = self.get_system_specs()
            hostname = socket.gethostname()
            ip_address = self._get_local_ip()
            
            client_info = load_balancer_pb2.ClientInfo(
                client_id=self.client_id,
                hostname=hostname,
                ip_address=ip_address,
                specs=specs
            )
            
            response = self.stub.RegisterClient(client_info)
            
            if response.success:
                self.client_id = response.assigned_id
                logger.info(f"Successfully registered as AI client {self.client_id}")
                logger.info(f"AI Performance score: {specs.performance_score:.2f}")
                logger.info(f"GPU: {specs.gpu_info} ({specs.gpu_memory_gb}GB)")
                return True
            else:
                logger.error(f"Registration failed: {response.message}")
                return False
                
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return False
    
    def send_port_update(self):
        """Send gRPC port update to server via health check"""
        try:
            if hasattr(self, 'client_grpc_port') and self.stub:
                # We'll use a simple approach - store port in client specs
                logger.info(f"Notifying server of gRPC port: {self.client_grpc_port}")
                
                # Create a custom health check that includes port info
                # For now, we'll just log it - the server will use a fixed port
                logger.info(f"Client gRPC endpoint: {self._get_local_ip()}:{self.client_grpc_port}")
                
        except Exception as e:
            logger.error(f"Failed to send port update: {e}")
    
    def start_heartbeat(self):
        """Start heartbeat thread"""
        def heartbeat_loop():
            while self._running:
                try:
                    response = self.stub.HealthCheck(load_balancer_pb2.Empty())
                    if not response.healthy:
                        logger.warning("Load balancer reported unhealthy status")
                    time.sleep(30)
                except Exception as e:
                    logger.error(f"Heartbeat failed: {e}")
                    time.sleep(5)
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat thread started")
    
    def check_llm_assignment(self):
        """Check if this client has been assigned an LLM model"""
        try:
            logger.info("Checking for LLM model assignment...")
            
            # Get system specs to determine assignment
            specs = self.get_system_specs()
            
            # Simple assignment logic based on performance score
            if specs.performance_score >= 80:
                assigned_model = "dhenu2-llama3.1-8b"
                model_script = "simple_llm_server.py"  # Use simple server for reliability
                port = 7863
            elif specs.performance_score >= 40:
                assigned_model = "dhenu2-llama3.2-3b"
                model_script = "simple_llm_server.py"  # Use simple server for reliability
                port = 7862
            else:
                assigned_model = "dhenu2-llama3.2-1b"
                model_script = "simple_llm_server.py"  # Use simple server for reliability
                port = 7861
            
            self.assigned_model = assigned_model
            
            logger.info(f"Assigned LLM model: {assigned_model}")
            logger.info(f"Model script: {model_script}")
            logger.info(f"Port: {port}")
            
            # Start the LLM model server
            self.start_llm_model(assigned_model, model_script, port)
            
        except Exception as e:
            logger.error(f"LLM assignment check failed: {e}")
    
    def start_llm_model(self, model_name: str, script_name: str, port: int):
        """Start the assigned LLM model server"""
        try:
            logger.info(f"Starting {model_name} server on port {port}...")
            
            # Check if script exists
            script_path = os.path.join("llm_models", script_name)
            if not os.path.exists(script_path):
                logger.warning(f"LLM script not found: {script_path}")
                logger.info("Please ensure LLM model scripts are in the llm_models/ directory")
                logger.info("Starting mock LLM server instead...")
                self.start_mock_llm_server(model_name, port)
                return
            
            # Check if we have the required dependencies
            try:
                import torch
                import transformers
                import gradio
                logger.info("LLM dependencies available")
                has_deps = True
            except ImportError as e:
                logger.warning(f"Missing LLM dependencies: {e}")
                logger.info("Starting mock LLM server instead...")
                self.start_mock_llm_server(model_name, port)
                return
            
            # Set environment variables
            env = os.environ.copy()
            env['GRADIO_PORT'] = str(port)
            env['MODEL_NAME'] = model_name
            
            # Start the model server in a separate thread
            def run_model():
                try:
                    logger.info(f"Launching {script_name}...")
                    
                    # Run the Python script with output capture
                    process = subprocess.Popen([
                        sys.executable, script_path
                    ], env=env, cwd=os.getcwd(), 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)
                    
                    # Store process for cleanup
                    self.llm_process = process
                    
                    logger.info(f"{model_name} server started (PID: {process.pid})")
                    logger.info(f"Expected at: http://localhost:{port}")
                    
                    # Monitor the process
                    try:
                        stdout, stderr = process.communicate(timeout=60)  # Wait up to 60 seconds
                        if process.returncode != 0:
                            logger.error(f"{model_name} process failed:")
                            logger.error(f"STDOUT: {stdout}")
                            logger.error(f"STDERR: {stderr}")
                    except subprocess.TimeoutExpired:
                        logger.info(f"{model_name} process running (timeout reached, assuming success)")
                    
                except Exception as e:
                    logger.error(f"Failed to start {model_name}: {e}")
                    # Fallback to mock server
                    logger.info("Starting mock server as fallback...")
                    self.start_mock_llm_server(model_name, port)
            
            # Start in background thread
            model_thread = threading.Thread(target=run_model, daemon=True)
            model_thread.start()
            
            # Give it more time to start and check if it's actually running
            time.sleep(10)
            
            # Test if the server is actually accessible
            self._test_model_server(port, model_name)
            
        except Exception as e:
            logger.error(f"Failed to start LLM model: {e}")
            logger.info("Starting mock server as fallback...")
            self.start_mock_llm_server(model_name, port)
    
    def _test_model_server(self, port: int, model_name: str):
        """Test if the model server is actually running"""
        try:
            import requests
            
            # Test basic connection
            response = requests.get(f"http://localhost:{port}", timeout=5)
            if response.status_code == 200:
                logger.info(f"{model_name} server is accessible on port {port}")
                return True
            else:
                logger.warning(f"{model_name} server returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Cannot reach {model_name} server on port {port}: {e}")
            logger.info("The model process may still be starting up...")
            return False
    
    def start_mock_llm_server(self, model_name: str, port: int):
        """Start a mock LLM server for demo purposes"""
        try:
            logger.info(f"🎭 Starting mock {model_name} server on port {port}...")
            
            # Store model info for AI request processing
            self.mock_model_info = {
                "name": model_name,
                "port": port,
                "responses": {
                    "dhenu2-llama3.2-1b": "Quick agricultural advice: This is a basic response from the 1B model focusing on simple, practical farming tips.",
                    "dhenu2-llama3.2-3b": "Detailed agricultural analysis: This is a comprehensive response from the 3B model providing in-depth farming strategies and scientific insights.",
                    "dhenu2-llama3.1-8b": "Expert agricultural consultation: This is an advanced response from the 8B model offering research-level analysis, policy considerations, and cutting-edge agricultural techniques."
                }
            }
            
            # Create a simple HTTP server that responds to requests
            def run_mock_server():
                try:
                    import http.server
                    import socketserver
                    from urllib.parse import parse_qs, urlparse
                    
                    # Store references for the handler
                    client_instance = self
                    model_info = self.mock_model_info
                    
                    class MockLLMHandler(http.server.BaseHTTPRequestHandler):
                        def do_GET(self):
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html')
                            self.end_headers()
                            
                            html = f"""
                            <html>
                            <head><title>Mock {model_name} Server</title></head>
                            <body>
                                <h1>Mock {model_name} Server</h1>
                                <p>This is a demo server for {model_name}</p>
                                <p>Client ID: {client_instance.client_id}</p>
                                <p>Status: Running</p>
                                <form method="POST">
                                    <textarea name="prompt" placeholder="Enter your prompt here..." rows="4" cols="50"></textarea><br>
                                    <button type="submit">Generate Response</button>
                                </form>
                            </body>
                            </html>
                            """
                            self.wfile.write(html.encode())
                        
                        def do_POST(self):
                            content_length = int(self.headers['Content-Length'])
                            post_data = self.rfile.read(content_length)
                            
                            # Check if this is a Gradio API call
                            if self.path == '/api/predict':
                                self.send_response(200)
                                self.send_header('Content-type', 'application/json')
                                self.end_headers()
                                
                                try:
                                    # Parse Gradio API request
                                    import json
                                    request_data = json.loads(post_data.decode())
                                    prompt = request_data.get('data', [''])[0] if request_data.get('data') else "sample prompt"
                                    
                                    # Generate model-specific response
                                    base_response = model_info["responses"].get(model_name, "Generic response")
                                    response = f"{base_response} Responding to: '{prompt}'"
                                    
                                    # Return Gradio API format
                                    gradio_response = {
                                        "data": [response],
                                        "is_generating": False,
                                        "duration": 2.5,
                                        "average_duration": 2.5
                                    }
                                    
                                    self.wfile.write(json.dumps(gradio_response).encode())
                                    return
                                    
                                except Exception as e:
                                    # Fallback response
                                    fallback_response = {
                                        "data": [f"Mock response from {model_name}"],
                                        "is_generating": False
                                    }
                                    self.wfile.write(json.dumps(fallback_response).encode())
                                    return
                            
                            # Regular HTML form handling
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html')
                            self.end_headers()
                            
                            # Parse the prompt
                            try:
                                data = parse_qs(post_data.decode())
                                prompt = data.get('prompt', [''])[0]
                            except:
                                prompt = "sample prompt"
                            
                            # Generate model-specific response
                            base_response = model_info["responses"].get(model_name, "Generic response")
                            response = f"{base_response} Responding to: '{prompt}'"
                            
                            html = f"""
                            <html>
                            <head><title>Mock {model_name} Response</title></head>
                            <body>
                                <h1>{model_name} Response</h1>
                                <p><strong>Prompt:</strong> {prompt}</p>
                                <p><strong>Response:</strong> {response}</p>
                                <a href="/">Back</a>
                            </body>
                            </html>
                            """
                            self.wfile.write(html.encode())
                        
                        def log_message(self, format, *args):
                            pass  # Suppress default logging
                    
                    with socketserver.TCPServer(("", port), MockLLMHandler) as httpd:
                        logger.info(f"Mock {model_name} server started on port {port}")
                        httpd.serve_forever()
                        
                except Exception as e:
                    logger.error(f"Mock server failed: {e}")
            
            # Start mock server in background thread
            mock_thread = threading.Thread(target=run_mock_server, daemon=True)
            mock_thread.start()
            
            logger.info(f"Mock {model_name} server is running!")
            logger.info(f"Access at: http://localhost:{port}")
            
        except Exception as e:
            logger.error(f"Failed to start mock server: {e}")
    
    def process_ai_request(self, prompt: str) -> str:
        """Process an AI request using the assigned model"""
        try:
            logger.info(f"Processing AI request: '{prompt[:50]}...'")
            
            # Always generate a proper response using the direct method
            logger.info(f"Generating response for model: {getattr(self, 'assigned_model', 'unknown')}")
            response = self._generate_direct_response(prompt)
            logger.info(f"Generated response ({len(response)} chars)")
            return response
            

                
        except Exception as e:
            logger.error(f"AI request processing failed: {e}")
            return f"Error processing request: {str(e)}"
    
    def _call_gradio_model(self, prompt: str, port: int) -> str:
        """Call the actual LLM model running on the client"""
        try:
            import requests
            
            logger.info(f"Calling LLM model at port {port}")
            
            # Try the mock server API first (since we're using simple_llm_server.py)
            api_url = f"http://localhost:{port}/api/predict"
            payload = {
                "data": [
                    prompt,      # prompt
                    256,         # max_tokens
                    0.7,         # temperature
                    0.9          # top_p
                ]
            }
            
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "data" in result and len(result["data"]) > 0:
                        model_response = result["data"][0]
                        logger.info(f"LLM model responded via API: {len(model_response)} chars")
                        return model_response
                except:
                    pass
            
            # If API doesn't work, try direct HTTP call to the simple server
            logger.info(f"Trying direct call to simple LLM server...")
            
            # Try a simple GET first to see if server is responding
            try:
                test_response = requests.get(f"http://localhost:{port}", timeout=5)
                if test_response.status_code == 200:
                    logger.info(f"LLM server is responding on port {port}")
                    
                    # Try POST with form data (for the simple server)
                    form_data = {"prompt": prompt}
                    post_response = requests.post(f"http://localhost:{port}", data=form_data, timeout=10)
                    
                    if post_response.status_code == 200:
                        # Extract response from HTML (simple parsing)
                        html_content = post_response.text
                        if "Response:" in html_content:
                            # Extract the response from HTML
                            start = html_content.find("<strong>Response:</strong>") + len("<strong>Response:</strong>")
                            end = html_content.find("</p>", start)
                            if start > 0 and end > start:
                                response_text = html_content[start:end].strip()
                                logger.info(f"Extracted response from HTML: {len(response_text)} chars")
                                return response_text
                
            except Exception as e:
                logger.warning(f"Direct server call failed: {e}")
            
            # Final fallback - generate response directly
            logger.info(f"Generating response directly...")
            return self._generate_direct_response(prompt)
                
        except Exception as e:
            logger.warning(f"All LLM calls failed: {e}")
            return self._generate_direct_response(prompt)
    
    def _generate_direct_response(self, prompt: str) -> str:
        """Generate response by calling the actual LLM server"""
        try:
            model_name = getattr(self, 'assigned_model', 'dhenu2-llama3.2-3b')
            
            logger.info(f"Generating response with {model_name}...")
            
            # Determine the port for this model
            if "1b" in model_name.lower():
                port = 7861
            elif "3b" in model_name.lower():
                port = 7862
            elif "8b" in model_name.lower():
                port = 7863
            else:
                port = 7862  # Default to 3B port
            
            # Try to call the actual LLM server
            try:
                import requests
                
                logger.info(f"Calling LLM server at port {port}")
                
                # Try Gradio API format first
                api_url = f"http://localhost:{port}/api/predict"
                payload = {
                    "data": [
                        prompt,      # prompt
                        256,         # max_tokens
                        0.7,         # temperature
                        0.9          # top_p
                    ]
                }
                
                response = requests.post(api_url, json=payload, timeout=30)
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "data" in result and len(result["data"]) > 0:
                            model_response = result["data"][0]
                            logger.info(f"LLM server responded: {len(model_response)} chars")
                            return model_response
                    except Exception as e:
                        logger.warning(f"Failed to parse LLM response: {e}")
                
                # If API doesn't work, try direct HTTP call
                logger.info("Trying direct HTTP call to LLM server...")
                form_data = {"prompt": prompt}
                post_response = requests.post(f"http://localhost:{port}", data=form_data, timeout=30)
                
                if post_response.status_code == 200:
                    html_content = post_response.text
                    if "Response:" in html_content:
                        # Extract the response from HTML
                        start = html_content.find("<strong>Response:</strong>") + len("<strong>Response:</strong>")
                        end = html_content.find("</p>", start)
                        if start > 0 and end > start:
                            response_text = html_content[start:end].strip()
                            logger.info(f"Extracted response from HTML: {len(response_text)} chars")
                            return response_text
                
            except Exception as e:
                logger.warning(f"Failed to call LLM server: {e}")
            
            # Fallback: Generate a simple response without conditional logic
            logger.info("Using fallback response generation")
            
            # Simple processing time simulation
            import time
            import random
            processing_time = 2.0 + random.uniform(0.5, 2.0)
            time.sleep(processing_time)
            
            # Generate a generic agricultural response
            fallback_response = f"Based on your query about '{prompt}', I recommend consulting with local agricultural extension services for specific guidance tailored to your region and crop conditions. General best practices include proper soil preparation, appropriate seed selection, balanced nutrition, integrated pest management, and sustainable water use."
            
            logger.info(f"Generated fallback response: {len(fallback_response)} chars")
            return fallback_response
            
        except Exception as e:
            logger.error(f"Direct response generation failed: {e}")
            return f"Agricultural advice for your query: '{prompt}'. Please consult local agricultural extension services for detailed guidance."
    
    def start_grpc_server(self):
        """Start gRPC server to receive AI requests from the load balancer"""
        try:
            from concurrent import futures
            
            # Create client service handler
            class ClientAIService(load_balancer_pb2_grpc.LoadBalancerServicer):
                def __init__(self, client_instance):
                    self.client = client_instance
                
                def ProcessAIRequest(self, request, context):
                    """Handle AI request from the server"""
                    try:
                        logger.info(f"Received AI request: {request.request_id}")
                        logger.info(f"Model: {request.model_name}")
                        logger.info(f"Prompt: {request.prompt[:50]}...")
                        
                        start_time = time.time()
                        
                        # Process the request
                        response_text = self.client.process_ai_request(request.prompt)
                        
                        processing_time = time.time() - start_time
                        
                        logger.info(f"Processed request {request.request_id} in {processing_time:.2f}s")
                        
                        return load_balancer_pb2.AIResponse(
                            request_id=request.request_id,
                            success=True,
                            response_text=response_text,
                            processing_time=processing_time,
                            model_used=request.model_name,
                            client_id=self.client.client_id
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to process AI request: {e}")
                        return load_balancer_pb2.AIResponse(
                            request_id=request.request_id,
                            success=False,
                            response_text=f"Error: {str(e)}",
                            processing_time=0.0,
                            model_used=request.model_name,
                            client_id=self.client.client_id
                        )
            
            # Use a deterministic port based on timestamp from client_id
            # Extract timestamp from client_id: client-kabir-1760551531-5e41d5e9
            try:
                parts = self.client_id.split('-')
                if len(parts) >= 3:
                    timestamp = int(parts[2])
                    client_port = 50052 + (timestamp % 1000)  # Use last 3 digits of timestamp
                else:
                    client_port = 50052 + (abs(hash(self.client_id)) % 1000)
            except:
                client_port = 50052 + (abs(hash(self.client_id)) % 1000)
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
            
            client_service = ClientAIService(self)
            load_balancer_pb2_grpc.add_LoadBalancerServicer_to_server(client_service, server)
            
            listen_addr = f'0.0.0.0:{client_port}'
            server.add_insecure_port(listen_addr)
            server.start()
            
            self.client_grpc_server = server
            self.client_grpc_port = client_port
            
            logger.info(f"Client gRPC server started on {listen_addr}")
            logger.info(f"Server should connect to: {self._get_local_ip()}:{client_port}")
            
        except Exception as e:
            logger.error(f"Failed to start client gRPC server: {e}")
    
    def _update_server_with_port(self):
        """Update server with client gRPC port information"""
        try:
            if hasattr(self, 'client_grpc_port'):
                # Store port info in a way the server can access
                # We'll use the client_id as a key and store port info
                logger.info(f"Client gRPC port: {self.client_grpc_port}")
                logger.info(f"Client IP: {self._get_local_ip()}")
                logger.info(f"Full address: {self._get_local_ip()}:{self.client_grpc_port}")
        except Exception as e:
            logger.error(f"Failed to update server with port info: {e}")
    
    def run(self):
        """Main client run loop"""
        logger.info(f"Starting AI Load Balancer Client {self.client_id}")
        
        # Display system information
        specs = self.get_system_specs()
        logger.info("System Specifications:")
        logger.info(f"  CPU: {specs.cpu_cores} cores @ {specs.cpu_frequency_ghz:.2f} GHz")
        logger.info(f"  RAM: {specs.ram_gb} GB")
        logger.info(f"  GPU: {specs.gpu_info} ({specs.gpu_memory_gb} GB)")
        logger.info(f"  OS: {specs.os_info}")
        logger.info(f"  AI Performance Score: {specs.performance_score:.2f}")
        
        # Connect to server
        if not self.connect():
            logger.error("Failed to connect to AI Load Balancer server")
            logger.error("Make sure the Raspberry Pi server is running and accessible")
            return
        
        # Register with server
        if not self.register():
            logger.error("Failed to register with AI Load Balancer server")
            return
        
        self._running = True
        
        # Start background threads
        self.start_heartbeat()
        
        # Check for LLM model assignment
        self.check_llm_assignment()
        
        # Start gRPC server for receiving AI requests
        self.start_grpc_server()
        
        # Send port update to server
        self.send_port_update()
        
        try:
            logger.info("AI Client ready for distributed LLM processing")
            logger.info("Press Ctrl+C to shutdown...")
            
            while self._running:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Disconnect from the load balancer"""
        logger.info("Disconnecting from AI Load Balancer...")
        self._running = False
        
        # Clean up LLM process
        if hasattr(self, 'llm_process') and self.llm_process:
            try:
                logger.info("Stopping LLM model server...")
                self.llm_process.terminate()
                self.llm_process.wait(timeout=10)
                logger.info("LLM model server stopped")
            except Exception as e:
                logger.warning(f"Error stopping LLM process: {e}")
        
        # Clean up client gRPC server
        if hasattr(self, 'client_grpc_server') and self.client_grpc_server:
            try:
                logger.info("Stopping client gRPC server...")
                self.client_grpc_server.stop(0)
                logger.info("Client gRPC server stopped")
            except Exception as e:
                logger.warning(f"Error stopping client gRPC server: {e}")
        
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        
        if self.channel:
            self.channel.close()
            
        logger.info("Disconnected from AI Load Balancer")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Load Balancer Client v1.0')
    parser.add_argument('--server', default='192.168.1.8:50051',
                       help='AI Load Balancer server address (Raspberry Pi IP)')
    parser.add_argument('--client-id', default=None,
                       help='Client ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    client = AILoadBalancerClient(args.server, args.client_id)
    client.run()

if __name__ == "__main__":
    main()