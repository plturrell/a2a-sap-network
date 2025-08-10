#!/usr/bin/env python3
"""Test script to verify SAP UI5 Fiori Portal functionality"""

import requests
import json

BASE_URL = "http://localhost:8090"

def test_portal():
    print("Testing A2A Developer Portal...")
    
    # Test 1: Check HTML loads
    print("\n1. Testing HTML page...")
    resp = requests.get(BASE_URL)
    assert resp.status_code == 200
    assert "sap-ui-bootstrap" in resp.text
    assert "ComponentContainer" in resp.text
    print("✓ HTML page loads correctly")
    
    # Test 2: Check Component.js
    print("\n2. Testing Component.js...")
    resp = requests.get(f"{BASE_URL}/static/Component.js")
    assert resp.status_code == 200
    assert "UIComponent.extend" in resp.text
    assert "getRouter().initialize()" in resp.text
    print("✓ Component.js loads correctly")
    
    # Test 3: Check manifest.json
    print("\n3. Testing manifest.json...")
    resp = requests.get(f"{BASE_URL}/static/manifest.json")
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["sap.app"]["id"] == "a2a.portal"
    assert "routing" in manifest["sap.ui5"]
    print("✓ manifest.json loads correctly")
    
    # Test 4: Check all views
    print("\n4. Testing views...")
    views = ["App", "Projects", "ProjectDetail", "AgentBuilder", "BPMNDesigner", "CodeEditor"]
    for view in views:
        resp = requests.get(f"{BASE_URL}/static/view/{view}.view.xml")
        assert resp.status_code == 200
        assert "mvc:View" in resp.text
        print(f"✓ {view}.view.xml loads correctly")
    
    # Test 5: Check all controllers
    print("\n5. Testing controllers...")
    for controller in views:
        resp = requests.get(f"{BASE_URL}/static/controller/{controller}.controller.js")
        assert resp.status_code == 200
        assert "Controller.extend" in resp.text
        print(f"✓ {controller}.controller.js loads correctly")
    
    # Test 6: Check i18n
    print("\n6. Testing i18n resources...")
    resp = requests.get(f"{BASE_URL}/static/i18n/i18n.properties")
    assert resp.status_code == 200
    assert "appTitle=" in resp.text
    print("✓ i18n.properties loads correctly")
    
    # Test 7: Check CSS
    print("\n7. Testing CSS...")
    resp = requests.get(f"{BASE_URL}/static/css/style.css")
    assert resp.status_code == 200
    assert "sapUiSizeCompact" in resp.text
    print("✓ style.css loads correctly")
    
    # Test 8: Test CRUD operations
    print("\n8. Testing CRUD operations...")
    
    # Create
    project_data = {
        "name": "Test Project",
        "description": "Testing CRUD operations"
    }
    resp = requests.post(f"{BASE_URL}/api/projects", json=project_data)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] == True
    project_id = data["project"]["project_id"]
    print("✓ Create project works")
    
    # Read
    resp = requests.get(f"{BASE_URL}/api/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["projects"]) > 0
    print("✓ Read projects works")
    
    # Update would go here if endpoint existed
    
    # Delete - endpoint not implemented yet
    # resp = requests.delete(f"{BASE_URL}/api/projects/{project_id}")
    
    print("\n✅ All tests passed! SAP UI5 Fiori Portal is working correctly.")
    print("\nSummary:")
    print("- All UI5 resources load properly")
    print("- All views and controllers are accessible")
    print("- Routing is configured correctly")
    print("- i18n and CSS resources are available")
    print("- Basic CRUD operations work (Create/Read)")
    print("\nThe portal is ready at: http://localhost:8090")

if __name__ == "__main__":
    try:
        test_portal()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to portal. Is it running on port 8090?")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")