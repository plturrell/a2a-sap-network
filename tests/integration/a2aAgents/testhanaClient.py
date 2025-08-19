#!/usr/bin/env python3
"""
Test HANA client to see what's wrong
"""
import sys
sys.path.append('/Users/apple/projects/finsight_cib/backend')

from app.clients.hana_client import get_hana_client, HANA_AVAILABLE

def test_hana_client():
    print(f"=== HANA CLIENT TEST ===")
    print(f"HANA Available: {HANA_AVAILABLE}")
    
    if not HANA_AVAILABLE:
        print("❌ HANA client library (hdbcli) not available")
        return
    
    try:
        hana_client = get_hana_client()
        print(f"✅ HANA client instance created: {type(hana_client)}")
        print(f"Client object: {hana_client}")
        
        # Check if client has the execute_query method
        if hasattr(hana_client, 'execute_query'):
            print("✅ execute_query method available")
        else:
            print("❌ execute_query method NOT available")
            print(f"Available methods: {[m for m in dir(hana_client) if not m.startswith('_')]}")
        
        # Try health check
        health = hana_client.health_check()
        print(f"Health check result: {health}")
        
    except Exception as e:
        print(f"❌ Error creating HANA client: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hana_client()