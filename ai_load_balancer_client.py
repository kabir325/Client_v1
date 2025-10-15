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
        
        logger.info(f"🚀 AI Load Balancer Client v1.0 initialized")
        logger.info(f"🆔 Client ID: {self.client_id}")
        logger.info(f"🌐 Server: {self.server_address}")
    
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
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"
    
    def connect(self) -> bool:
        """Connect to the AI load balancer server"""
        try:
            logger.info(f"🔌 Connecting to AI Load Balancer at {self.server_address}...")
            
            self.channel = grpc.insecure_channel(self.server_address)
            self.stub = load_balancer_pb2_grpc.LoadBalancerStub(self.channel)
            
            # Test connection with health check
            response = self.stub.HealthCheck(load_balancer_pb2.Empty(), timeout=10)
            
            if response.healthy:
                logger.info("✅ Connected to AI Load Balancer successfully")
                return True
            else:
                logger.error("❌ Load balancer reported unhealthy status")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
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
                logger.info(f"✅ Successfully registered as AI client {self.client_id}")
                logger.info(f"🎯 AI Performance score: {specs.performance_score:.2f}")
                logger.info(f"🖥️ GPU: {specs.gpu_info} ({specs.gpu_memory_gb}GB)")
                return True
            else:
                logger.error(f"Registration failed: {response.message}")
                return False
                
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return False
    
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
        logger.info("💓 Heartbeat thread started")
    
    def check_llm_assignment(self):
        """Check if this client has been assigned an LLM model"""
        try:
            logger.info("🔍 Checking for LLM model assignment...")
            
            # Get system specs to determine assignment
            specs = self.get_system_specs()
            
            # Simple assignment logic based on performance score
            if specs.performance_score >= 80:
                assigned_model = "dhenu2-llama3.1-8b"
                model_script = "llama8B.py"
                port = 7863
            elif specs.performance_score >= 60:
                assigned_model = "dhenu2-llama3.2-3b"
                model_script = "llama3B.py"
                port = 7862
            else:
                assigned_model = "dhenu2-llama3.2-1b"
                model_script = "llama1B.py"
                port = 7861
            
            self.assigned_model = assigned_model
            
            logger.info(f"🎯 Assigned LLM model: {assigned_model}")
            logger.info(f"📝 Model script: {model_script}")
            logger.info(f"🌐 Port: {port}")
            
            # Start the LLM model server
            self.start_llm_model(assigned_model, model_script, port)
            
        except Exception as e:
            logger.error(f"❌ LLM assignment check failed: {e}")
    
    def start_llm_model(self, model_name: str, script_name: str, port: int):
        """Start the assigned LLM model server"""
        try:
            logger.info(f"🚀 Starting {model_name} server on port {port}...")
            
            # Check if script exists
            script_path = os.path.join("llm_models", script_name)
            if not os.path.exists(script_path):
                logger.warning(f"⚠️ LLM script not found: {script_path}")
                logger.info("💡 Please ensure LLM model scripts are in the llm_models/ directory")
                logger.info("🔧 For demo purposes, creating a mock LLM server...")
                self.start_mock_llm_server(model_name, port)
                return
            
            # Set environment variables
            env = os.environ.copy()
            env['GRADIO_PORT'] = str(port)
            env['MODEL_NAME'] = model_name
            
            # Start the model server in a separate thread
            def run_model():
                try:
                    logger.info(f"🔄 Launching {script_name}...")
                    
                    # Run the Python script
                    process = subprocess.Popen([
                        sys.executable, script_path
                    ], env=env, cwd=os.getcwd())
                    
                    # Store process for cleanup
                    self.llm_process = process
                    
                    logger.info(f"✅ {model_name} server started (PID: {process.pid})")
                    logger.info(f"🌐 Access at: http://localhost:{port}")
                    
                    # Wait for process to complete
                    process.wait()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to start {model_name}: {e}")
            
            # Start in background thread
            model_thread = threading.Thread(target=run_model, daemon=True)
            model_thread.start()
            
            # Give it time to start
            time.sleep(5)
            
            logger.info(f"🎉 {model_name} is now available for inference!")
            
        except Exception as e:
            logger.error(f"❌ Failed to start LLM model: {e}")
    
    def start_mock_llm_server(self, model_name: str, port: int):
        """Start a mock LLM server for demo purposes"""
        try:
            logger.info(f"🎭 Starting mock {model_name} server on port {port}...")
            
            # Create a simple HTTP server that responds to requests
            def run_mock_server():
                try:
                    import http.server
                    import socketserver
                    from urllib.parse import parse_qs, urlparse
                    
                    class MockLLMHandler(http.server.BaseHTTPRequestHandler):
                        def do_GET(self):
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html')
                            self.end_headers()
                            
                            html = f"""
                            <html>
                            <head><title>Mock {model_name} Server</title></head>
                            <body>
                                <h1>🤖 Mock {model_name} Server</h1>
                                <p>This is a demo server for {model_name}</p>
                                <p>Client ID: {self.client_id}</p>
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
                            
                            self.send_response(200)
                            self.send_header('Content-type', 'text/html')
                            self.end_headers()
                            
                            # Parse the prompt
                            try:
                                data = parse_qs(post_data.decode())
                                prompt = data.get('prompt', [''])[0]
                            except:
                                prompt = "sample prompt"
                            
                            response = f"[DEMO] Mock response from {model_name}: This is a simulated agricultural AI response to your prompt: '{prompt}'. In a real deployment, this would be processed by the actual LLM model."
                            
                            html = f"""
                            <html>
                            <head><title>Mock {model_name} Response</title></head>
                            <body>
                                <h1>🤖 {model_name} Response</h1>
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
                        logger.info(f"✅ Mock {model_name} server started on port {port}")
                        httpd.serve_forever()
                        
                except Exception as e:
                    logger.error(f"❌ Mock server failed: {e}")
            
            # Start mock server in background thread
            mock_thread = threading.Thread(target=run_mock_server, daemon=True)
            mock_thread.start()
            
            logger.info(f"🎉 Mock {model_name} server is running!")
            logger.info(f"🌐 Access at: http://localhost:{port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start mock server: {e}")
    
    def run(self):
        """Main client run loop"""
        logger.info(f"🚀 Starting AI Load Balancer Client {self.client_id}")
        
        # Display system information
        specs = self.get_system_specs()
        logger.info("🖥️ System Specifications:")
        logger.info(f"  CPU: {specs.cpu_cores} cores @ {specs.cpu_frequency_ghz:.2f} GHz")
        logger.info(f"  RAM: {specs.ram_gb} GB")
        logger.info(f"  GPU: {specs.gpu_info} ({specs.gpu_memory_gb} GB)")
        logger.info(f"  OS: {specs.os_info}")
        logger.info(f"  AI Performance Score: {specs.performance_score:.2f}")
        
        # Connect to server
        if not self.connect():
            logger.error("❌ Failed to connect to AI Load Balancer server")
            logger.error("💡 Make sure the Raspberry Pi server is running and accessible")
            return
        
        # Register with server
        if not self.register():
            logger.error("❌ Failed to register with AI Load Balancer server")
            return
        
        self._running = True
        
        # Start background threads
        self.start_heartbeat()
        
        # Check for LLM model assignment
        self.check_llm_assignment()
        
        try:
            logger.info("🎯 AI Client ready for distributed LLM processing")
            logger.info("Press Ctrl+C to shutdown...")
            
            while self._running:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Disconnect from the load balancer"""
        logger.info("🔌 Disconnecting from AI Load Balancer...")
        self._running = False
        
        # Clean up LLM process
        if hasattr(self, 'llm_process') and self.llm_process:
            try:
                logger.info("🛑 Stopping LLM model server...")
                self.llm_process.terminate()
                self.llm_process.wait(timeout=10)
                logger.info("✅ LLM model server stopped")
            except Exception as e:
                logger.warning(f"⚠️ Error stopping LLM process: {e}")
        
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        
        if self.channel:
            self.channel.close()
            
        logger.info("✅ Disconnected from AI Load Balancer")

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