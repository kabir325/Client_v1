#!/usr/bin/env python3
"""
Start AI Load Balancer Client v1.0
Simple startup script for laptop clients
"""

import os
import sys
import subprocess
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import grpc
        import load_balancer_pb2
        import load_balancer_pb2_grpc
        logger.info("✅ All requirements satisfied")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing requirement: {e}")
        logger.info("💡 Run: pip install -r requirements.txt")
        return False

def generate_grpc_files():
    """Generate gRPC files if they don't exist"""
    if not os.path.exists("load_balancer_pb2.py"):
        logger.info("🔧 Generating gRPC files...")
        try:
            subprocess.run([
                sys.executable, "-m", "grpc_tools.protoc",
                "--proto_path=.",
                "--python_out=.",
                "--grpc_python_out=.",
                "load_balancer.proto"
            ], check=True)
            logger.info("✅ gRPC files generated")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate gRPC files: {e}")
            return False
    return True

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='AI Load Balancer Client v1.0 Startup')
    parser.add_argument('--server', default='192.168.1.8:50051',
                       help='AI Load Balancer server address (Raspberry Pi IP)')
    parser.add_argument('--client-id', default=None,
                       help='Client ID (default: auto-generated)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting AI Load Balancer Client v1.0")
    logger.info("=" * 50)
    logger.info(f"🌐 Server: {args.server}")
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements not met")
        return False
    
    # Generate gRPC files
    if not generate_grpc_files():
        logger.error("❌ gRPC file generation failed")
        return False
    
    # Start the client
    logger.info("🎯 Starting client...")
    try:
        from ai_load_balancer_client import AILoadBalancerClient
        client = AILoadBalancerClient(args.server, args.client_id)
        client.run()
    except KeyboardInterrupt:
        logger.info("🛑 Client stopped by user")
    except Exception as e:
        logger.error(f"❌ Client error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)