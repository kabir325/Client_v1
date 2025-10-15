#!/usr/bin/env python3
"""
Check What Ports Are Actually Running
"""

import subprocess
import sys

def check_ports():
    """Check what ports are listening"""
    print("🔍 Checking what ports are listening...")
    
    try:
        # Windows netstat command
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            
            # Look for ports in the 7860-7870 range and 50000+ range
            relevant_ports = []
            for line in lines:
                if 'LISTENING' in line:
                    if ':786' in line or ':50' in line:
                        relevant_ports.append(line.strip())
            
            if relevant_ports:
                print("📊 Relevant listening ports:")
                for port in relevant_ports:
                    print(f"   {port}")
            else:
                print("⚠️ No relevant ports found listening")
                
            # Specifically check for Gradio ports
            gradio_ports = [7860, 7861, 7862, 7863]
            print(f"\n🔍 Checking specific Gradio ports:")
            for port in gradio_ports:
                found = any(f':{port}' in line for line in lines if 'LISTENING' in line)
                status = "✅ LISTENING" if found else "❌ NOT LISTENING"
                print(f"   Port {port}: {status}")
                
        else:
            print(f"❌ netstat failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking ports: {e}")

def check_processes():
    """Check for Python processes that might be running models"""
    print(f"\n🔍 Checking for Python processes...")
    
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            python_processes = [line for line in lines if 'python.exe' in line]
            
            if python_processes:
                print("🐍 Python processes running:")
                for proc in python_processes:
                    if proc.strip():
                        print(f"   {proc.strip()}")
            else:
                print("⚠️ No Python processes found")
        else:
            print(f"❌ tasklist failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")

if __name__ == "__main__":
    check_ports()
    check_processes()