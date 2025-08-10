#!/usr/bin/env python3
"""
Test script to verify CLI agent startup works
"""

import asyncio
import sys
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

async def test_agent_startup():
    """Test starting agent via CLI functionality"""
    try:
        from app.a2a.cli import start_agent
        
        print("🔄 Testing agent startup via CLI...")
        
        # Create a task to start the agent
        startup_task = asyncio.create_task(start_agent("agent0", port=8000))
        
        # Let it run for a few seconds to ensure it starts
        await asyncio.sleep(3)
        
        # Cancel the startup task
        startup_task.cancel()
        
        try:
            await startup_task
        except asyncio.CancelledError:
            print("✅ Agent startup test completed successfully!")
            print("   Agent started without errors and was cleanly cancelled")
            return True
        
    except Exception as e:
        print(f"❌ Agent startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the CLI test"""
    print("🚀 Starting A2A CLI Test Suite")
    print("=" * 50)
    
    success = await test_agent_startup()
    
    print("=" * 50)
    if success:
        print("🎉 CLI test passed!")
        return 0
    else:
        print("💥 CLI test failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)