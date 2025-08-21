#!/usr/bin/env python3
"""
A2A System Health Check Script
Checks all components for 100% system health
"""

import requests
import subprocess
import json
import sys
from datetime import datetime
import time

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_service(name, url, expected_status=200):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            return True, f"{Colors.GREEN}✓ {name}{Colors.END}"
        else:
            return False, f"{Colors.RED}✗ {name} (status: {response.status_code}){Colors.END}"
    except requests.exceptions.ConnectionError:
        return False, f"{Colors.RED}✗ {name} (not running){Colors.END}"
    except Exception as e:
        return False, f"{Colors.RED}✗ {name} (error: {str(e)}){Colors.END}"

def check_port(name, port):
    """Check if a port is listening"""
    result = subprocess.run(['lsof', '-i', f':{port}'], capture_output=True, text=True)
    if result.returncode == 0:
        return True, f"{Colors.GREEN}✓ {name} (port {port}){Colors.END}"
    else:
        return False, f"{Colors.RED}✗ {name} (port {port} not listening){Colors.END}"

def check_blockchain():
    """Check blockchain connectivity"""
    try:
        response = requests.post(
            'http://localhost:8545',
            json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1},
            timeout=5
        )
        if response.status_code == 200:
            block_number = int(response.json()['result'], 16)
            return True, f"{Colors.GREEN}✓ Blockchain (block #{block_number}){Colors.END}"
        else:
            return False, f"{Colors.RED}✗ Blockchain (invalid response){Colors.END}"
    except:
        return False, f"{Colors.RED}✗ Blockchain (not running){Colors.END}"

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}A2A System Health Check{Colors.END}")
    print(f"{Colors.BLUE}{'='*50}{Colors.END}\n")
    
    # Track overall health
    total_services = 0
    healthy_services = 0
    
    # Core Infrastructure (Priority 1)
    print(f"{Colors.BOLD}Core Infrastructure:{Colors.END}")
    services = [
        ("Blockchain (Anvil)", lambda: check_blockchain()),
        ("Network Service (CAP/CDS)", lambda: check_service("CAP Server", "http://localhost:4004/health")),
        ("Agents Service API", lambda: check_service("Agents API", "http://localhost:8000/health")),
    ]
    
    for name, check_func in services:
        total_services += 1
        is_healthy, message = check_func()
        if is_healthy:
            healthy_services += 1
        print(f"  {message}")
    
    # Infrastructure Services (Priority 2)
    print(f"\n{Colors.BOLD}Infrastructure Services:{Colors.END}")
    infrastructure = [
        ("Redis Cache", lambda: check_port("Redis", 6379)),
        ("Prometheus", lambda: check_service("Prometheus", "http://localhost:9090/-/healthy")),
    ]
    
    for name, check_func in infrastructure:
        total_services += 1
        is_healthy, message = check_func()
        if is_healthy:
            healthy_services += 1
        print(f"  {message}")
    
    # Individual Agents (Priority 3)
    print(f"\n{Colors.BOLD}A2A Agents (16 total):{Colors.END}")
    agents = [
        ("Data Product Agent", 8001),
        ("Data Standardization Agent", 8002),
        ("AI Preparation Agent", 8003),
        ("Vector Processing Agent", 8004),
        ("Calc Validation Agent", 8005),
        ("QA Validation Agent", 8006),
        ("Quality Control Manager", 8007),
        ("Reasoning Agent", 8008),
        ("SQL Agent", 8009),
        ("Agent Manager", 8010),
        ("Data Manager", 8011),
        ("Catalog Manager", 8012),
        ("Calculation Agent", 8013),
        ("Agent Builder", 8014),
        ("Embedding Fine-Tuner", 8015),
        ("Registry Server", 8000),
    ]
    
    for agent_name, port in agents:
        total_services += 1
        is_healthy, message = check_port(agent_name, port)
        if is_healthy:
            healthy_services += 1
        print(f"  {message}")
    
    # MCP Services (Priority 4)
    print(f"\n{Colors.BOLD}MCP Services:{Colors.END}")
    mcp_services = [
        ("Enhanced Test Suite", 8100),
        ("Data Standardization MCP", 8101),
        ("Vector Similarity", 8102),
        ("Vector Ranking", 8103),
        ("Transport Layer", 8104),
        ("Reasoning Agent MCP", 8105),
        ("Session Management", 8106),
        ("Resource Streaming", 8107),
        ("Confidence Calculator", 8108),
        ("Semantic Similarity", 8109),
    ]
    
    for service_name, port in mcp_services:
        total_services += 1
        is_healthy, message = check_port(service_name, port)
        if is_healthy:
            healthy_services += 1
        print(f"  {message}")
    
    # Calculate health percentage
    health_percentage = (healthy_services / total_services) * 100
    
    # Summary
    print(f"\n{Colors.BOLD}System Health Summary:{Colors.END}")
    print(f"  Total Services: {total_services}")
    print(f"  Healthy: {Colors.GREEN}{healthy_services}{Colors.END}")
    print(f"  Unhealthy: {Colors.RED}{total_services - healthy_services}{Colors.END}")
    
    # Health rating
    if health_percentage == 100:
        status = f"{Colors.GREEN}✓ FULLY OPERATIONAL (100%){Colors.END}"
    elif health_percentage >= 80:
        status = f"{Colors.GREEN}✓ OPERATIONAL ({health_percentage:.1f}%){Colors.END}"
    elif health_percentage >= 60:
        status = f"{Colors.YELLOW}⚠ PARTIALLY OPERATIONAL ({health_percentage:.1f}%){Colors.END}"
    else:
        status = f"{Colors.RED}✗ DEGRADED ({health_percentage:.1f}%){Colors.END}"
    
    print(f"\n{Colors.BOLD}Overall Status: {status}{Colors.END}")
    
    # Recommendations
    if health_percentage < 100:
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        if health_percentage < 30:
            print(f"  • Run: ./start.sh complete")
        elif health_percentage < 60:
            print(f"  • Check logs in /logs directory")
            print(f"  • Review agent startup errors")
        else:
            print(f"  • Check individual service logs")
            print(f"  • Verify all dependencies are installed")
    
    return 0 if health_percentage == 100 else 1

if __name__ == "__main__":
    sys.exit(main())