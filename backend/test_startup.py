#!/usr/bin/env python3
"""Test script to verify the FastAPI application can start"""
import sys
import asyncio
import uvicorn
from contextlib import asynccontextmanager

# Add current directory to path
sys.path.insert(0, '.')

print("Testing FastAPI application startup...")
print("-" * 50)

try:
    # Import the FastAPI app
    from main import app
    print("✓ FastAPI app imported successfully")
    
    # Test that we can access the app object
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    
    # List all routes
    print("\n✓ Registered routes:")
    routes = []
    for route in app.routes:
        if hasattr(route, 'path'):
            routes.append(f"  - {route.path}")
    
    # Show first 10 routes
    for route in sorted(routes)[:10]:
        print(route)
    print(f"  ... and {len(routes) - 10} more routes" if len(routes) > 10 else "")
    
    print("\n✓ All imports successful!")
    print("✓ Application is ready to start")
    print("\nTo run the application, use:")
    print("  uvicorn main:app --reload")
    
except Exception as e:
    print(f"\n✗ Error during startup test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)