#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

print(f"XAI_API_KEY exists: {os.getenv('XAI_API_KEY') is not None}")
print(f"XAI_API_KEY length: {len(os.getenv('XAI_API_KEY', ''))}")
print(f"XAI_BASE_URL: {os.getenv('XAI_BASE_URL', 'Not set')}")
print(f"XAI_MODEL: {os.getenv('XAI_MODEL', 'Not set')}")

# Check if the key looks valid (starts with expected pattern)
key = os.getenv('XAI_API_KEY', '')
if key:
    print(f"API Key starts with: {key[:10]}...")