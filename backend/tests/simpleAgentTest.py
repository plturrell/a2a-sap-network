#!/usr/bin/env python3
"""
Simple agent test without complex imports
"""

print("Creating a minimal agent test...")

# Test basic FastAPI functionality
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    
    app = FastAPI(title="Test Agent", version="1.0.0")
    
    @app.get("/")
    async def root():
        return {"message": "Test agent is running!", "status": "ok"}
    
    @app.get("/health")
    async def health():
        return JSONResponse(
            content={
                "status": "healthy",
                "agent": "test_agent",
                "version": "1.0.0"
            }
        )
    
    print("✅ FastAPI app created successfully!")
    print("✅ Basic agent structure is working!")
    
    # Test can create the app
    print("\nTo run this test agent:")
    print("  python3 simple_agent_test.py --run")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("\n🚀 Starting test agent on http://localhost:8001")
        print("Press CTRL+C to stop")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()