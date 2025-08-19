#!/usr/bin/env python3
"""
Test Agent 0 with Dublin Core Enhancement
"""

import requests
import json
import time

def test_agent0_dublin_core():
    """Test Agent 0 with Dublin Core metadata extraction and quality assessment"""
    
    # Agent 0 endpoints
    agent0_url = "http://localhost:8000/a2a/agent0/v1"
    
    print("Testing Agent 0 with Dublin Core Enhancement")
    print("=" * 60)
    
    # Get agent card
    print("\n1. Getting Agent Card...")
    response = requests.get(f"{agent0_url}/.well-known/agent.json")
    if response.status_code == 200:
        agent_card = response.json()
        print(f"✓ Agent: {agent_card['name']}")
        print(f"  Version: {agent_card['version']}")
        print(f"  Dublin Core Compliance: {agent_card['capabilities'].get('dublinCoreCompliance', False)}")
        
        # Check for Dublin Core skills
        dublin_core_skills = [s for s in agent_card['skills'] if 'dublin-core' in str(s).lower()]
        print(f"\n  Dublin Core Skills Found: {len(dublin_core_skills)}")
        for skill in dublin_core_skills[:3]:
            print(f"    - {skill['name']}: {skill['description']}")
    else:
        print(f"✗ Failed to get agent card: {response.status_code}")
        return
    
    # Send message to process data with Dublin Core
    print("\n\n2. Sending Data Processing Request...")
    message = {
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Process raw financial data with Dublin Core metadata extraction"
                },
                {
                    "kind": "data",
                    "data": {
                        "processing_instructions": {
                            "action": "register_data_products",
                            "enable_dublin_core": True,
                            "quality_threshold": 0.6
                        }
                    }
                }
            ]
        },
        "contextId": f"test_dublin_core_{int(time.time())}"
    }
    
    response = requests.post(f"{agent0_url}/messages", json=message)
    if response.status_code == 200:
        result = response.json()
        task_id = result['taskId']
        print(f"✓ Task created: {task_id}")
        
        # Poll for task completion
        print("\n3. Monitoring Task Progress...")
        for i in range(30):  # Poll for up to 30 seconds
            time.sleep(1)
            
            status_response = requests.get(f"{agent0_url}/tasks/{task_id}")
            if status_response.status_code == 200:
                task_status = status_response.json()
                current_state = task_status['status']['state']
                
                if current_state == "completed":
                    print(f"\n✓ Task completed successfully!")
                    
                    # Check for Dublin Core in artifacts
                    if task_status.get('artifacts'):
                        artifact = task_status['artifacts'][0]
                        data = artifact['parts'][0]['data']
                        
                        # Display Dublin Core metadata
                        if 'dublin_core_metadata' in data:
                            dc = data['dublin_core_metadata']
                            print(f"\n4. Dublin Core Metadata Extracted:")
                            print(f"   Title: {dc.get('title', 'N/A')}")
                            print(f"   Creator: {', '.join(dc.get('creator', []))}")
                            print(f"   Subject: {', '.join(dc.get('subject', [])[:5])}")
                            print(f"   Publisher: {dc.get('publisher', 'N/A')}")
                            print(f"   Type: {dc.get('type', 'N/A')}")
                            print(f"   Format: {dc.get('format', 'N/A')}")
                            print(f"   Language: {dc.get('language', 'N/A')}")
                            print(f"   Rights: {dc.get('rights', 'N/A')}")
                        
                        # Display quality assessment
                        if 'dublin_core_quality' in data:
                            quality = data['dublin_core_quality']
                            print(f"\n5. Dublin Core Quality Assessment:")
                            print(f"   Overall Score: {quality.get('overall_score', 0):.2f}")
                            print(f"   Completeness: {quality.get('completeness', 0):.2%}")
                            print(f"   Populated Elements: {quality.get('populated_elements', 0)}/{quality.get('total_elements', 15)}")
                            
                            compliance = quality.get('standards_compliance', {})
                            print(f"\n   Standards Compliance:")
                            print(f"   - ISO 15836: {'✓' if compliance.get('iso15836_compliant') else '✗'}")
                            print(f"   - RFC 5013: {'✓' if compliance.get('rfc5013_compliant') else '✗'}")
                            print(f"   - ANSI/NISO: {'✓' if compliance.get('ansi_niso_compliant') else '✗'}")
                            
                            if quality.get('recommendations'):
                                print(f"\n   Recommendations:")
                                for rec in quality['recommendations']:
                                    print(f"   - {rec}")
                        
                        # Check ORD descriptor
                        if 'ord_descriptor' in data:
                            ord = data['ord_descriptor']
                            if 'dublinCore' in ord:
                                print(f"\n6. ORD Document includes Dublin Core: ✓")
                                print(f"   Data Products with Dublin Core: {len([dp for dp in ord.get('dataProducts', []) if 'dublinCore' in dp])}")
                        
                        # Check downstream trigger
                        if 'downstream_trigger' in data:
                            trigger = data['downstream_trigger']
                            if 'message' in trigger and 'dublin_core_context' in trigger['message']:
                                dc_context = trigger['message']['dublin_core_context']
                                print(f"\n7. Dublin Core Context Passed to Downstream Agent: ✓")
                                print(f"   Quality Score: {dc_context.get('quality_score', 0):.2f}")
                                print(f"   Standards Compliant: {dc_context.get('standards_compliant', False)}")
                                print(f"   Completeness: {dc_context.get('completeness', 0):.2%}")
                    
                    break
                    
                elif current_state == "failed":
                    print(f"\n✗ Task failed")
                    if task_status['status'].get('error'):
                        print(f"   Error: {task_status['status']['error']}")
                    break
                else:
                    print(f"\r   Status: {current_state} {'.' * (i % 4)}    ", end='', flush=True)
            else:
                print(f"\n✗ Failed to get task status: {status_response.status_code}")
                break
                
    else:
        print(f"✗ Failed to create task: {response.status_code}")
        print(f"   Error: {response.text}")


if __name__ == "__main__":
    test_agent0_dublin_core()
    print("\n\n✓ Agent 0 Dublin Core test completed!")