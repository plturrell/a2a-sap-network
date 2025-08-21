#!/usr/bin/env python3
"""
Launcher for Enhanced A2A Test Suite MCP Server
Uses Python 3.11 with proper MCP library support
"""

import subprocess
import sys
import os
from pathlib import Path

def start_enhanced_mcp_server():
    """Start the Enhanced MCP server with Python 3.11"""
    
    # Get the path to the enhanced MCP server
    current_dir = Path(__file__).parent
    server_path = current_dir / "enhanced_mcp_server.py"
    
    # Use Python 3.11 to run the server
    python311_path = "/opt/homebrew/bin/python3.11"
    
    if not Path(python311_path).exists():
        print("Error: Python 3.11 not found at expected path")
        print("Please install Python 3.11 or update the path in this script")
        return False
    
    try:
        print("🚀 Starting Enhanced A2A Test Suite MCP Server...")
        print(f"   • Using Python 3.11: {python311_path}")
        print(f"   • Server script: {server_path}")
        print(f"   • Port: 8100")
        
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(current_dir.parent)
        
        # Start the server as a subprocess
        process = subprocess.Popen([
            python311_path,
            str(server_path),
            "--port", "8100"
        ], env=env)
        
        print(f"   • Process ID: {process.pid}")
        print("✅ Enhanced MCP Server started successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Enhanced MCP Server: {e}")
        return False

if __name__ == "__main__":
    success = start_enhanced_mcp_server()
    if not success:
        sys.exit(1)