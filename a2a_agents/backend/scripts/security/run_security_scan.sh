#!/bin/bash

# Security Scanning Script for A2A Platform
# This script runs various security scans required before production deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORTS_DIR="${PROJECT_ROOT}/security-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create reports directory
mkdir -p "${REPORTS_DIR}"

echo "=========================================="
echo "A2A Platform Security Scanning"
echo "Timestamp: ${TIMESTAMP}"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Dependency Vulnerability Scanning
echo -e "\n${YELLOW}[1/6] Running dependency vulnerability scan...${NC}"
if command_exists safety; then
    safety check --json > "${REPORTS_DIR}/dependency-scan-${TIMESTAMP}.json" || true
    echo -e "${GREEN}✓ Dependency scan completed${NC}"
else
    echo -e "${RED}✗ Safety not installed. Run: pip install safety${NC}"
fi

# 2. Static Application Security Testing (SAST)
echo -e "\n${YELLOW}[2/6] Running static code analysis...${NC}"
if command_exists bandit; then
    bandit -r "${PROJECT_ROOT}/backend" -f json -o "${REPORTS_DIR}/sast-bandit-${TIMESTAMP}.json" || true
    echo -e "${GREEN}✓ SAST scan completed${NC}"
else
    echo -e "${RED}✗ Bandit not installed. Run: pip install bandit${NC}"
fi

# 3. Secret Detection
echo -e "\n${YELLOW}[3/6] Scanning for secrets...${NC}"
if command_exists trufflehog; then
    trufflehog filesystem "${PROJECT_ROOT}" --json > "${REPORTS_DIR}/secrets-scan-${TIMESTAMP}.json" || true
    echo -e "${GREEN}✓ Secret scanning completed${NC}"
else
    echo -e "${RED}✗ Trufflehog not installed. See: https://github.com/trufflesecurity/trufflehog${NC}"
fi

# 4. Container Security Scanning
echo -e "\n${YELLOW}[4/6] Scanning Docker images...${NC}"
if [ -f "${PROJECT_ROOT}/backend/Dockerfile" ]; then
    if command_exists trivy; then
        trivy fs "${PROJECT_ROOT}/backend" --format json --output "${REPORTS_DIR}/container-scan-${TIMESTAMP}.json" || true
        echo -e "${GREEN}✓ Container scan completed${NC}"
    else
        echo -e "${RED}✗ Trivy not installed. See: https://github.com/aquasecurity/trivy${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No Dockerfile found${NC}"
fi

# 5. Security Headers Check
echo -e "\n${YELLOW}[5/6] Checking security headers configuration...${NC}"
python3 - << EOF
import json
import re

security_headers = [
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "Referrer-Policy",
    "Permissions-Policy"
]

findings = []
code_files = []

# Search for security header implementations
import os
for root, dirs, files in os.walk("${PROJECT_ROOT}/backend"):
    for file in files:
        if file.endswith(('.py', '.js', '.ts')):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for header in security_headers:
                    if header in content:
                        code_files.append({
                            "file": filepath,
                            "header": header
                        })

report = {
    "scan_type": "security_headers",
    "timestamp": "${TIMESTAMP}",
    "required_headers": security_headers,
    "implementations_found": code_files
}

with open("${REPORTS_DIR}/headers-check-${TIMESTAMP}.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Found {len(code_files)} security header implementations")
EOF
echo -e "${GREEN}✓ Security headers check completed${NC}"

# 6. OWASP Dependency Check
echo -e "\n${YELLOW}[6/6] Running OWASP dependency check...${NC}"
if [ -f "${PROJECT_ROOT}/backend/requirements.txt" ]; then
    pip-audit --desc --format json --output "${REPORTS_DIR}/owasp-dependencies-${TIMESTAMP}.json" || true
    echo -e "${GREEN}✓ OWASP dependency check completed${NC}"
else
    echo -e "${YELLOW}⚠ No requirements.txt found${NC}"
fi

# Generate summary report
echo -e "\n${YELLOW}Generating summary report...${NC}"
python3 - << EOF
import json
import os
from datetime import datetime

summary = {
    "scan_date": "${TIMESTAMP}",
    "scan_results": {
        "dependency_scan": os.path.exists("${REPORTS_DIR}/dependency-scan-${TIMESTAMP}.json"),
        "sast_scan": os.path.exists("${REPORTS_DIR}/sast-bandit-${TIMESTAMP}.json"),
        "secrets_scan": os.path.exists("${REPORTS_DIR}/secrets-scan-${TIMESTAMP}.json"),
        "container_scan": os.path.exists("${REPORTS_DIR}/container-scan-${TIMESTAMP}.json"),
        "headers_check": os.path.exists("${REPORTS_DIR}/headers-check-${TIMESTAMP}.json"),
        "owasp_check": os.path.exists("${REPORTS_DIR}/owasp-dependencies-${TIMESTAMP}.json")
    },
    "recommendations": [
        "Schedule professional penetration testing before production deployment",
        "Conduct security architecture review with SAP security team",
        "Complete security training for development team",
        "Implement security monitoring and alerting",
        "Document incident response procedures"
    ]
}

with open("${REPORTS_DIR}/security-scan-summary-${TIMESTAMP}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSecurity scan summary saved to: ${REPORTS_DIR}/security-scan-summary-${TIMESTAMP}.json")
EOF

echo -e "\n${GREEN}=========================================="
echo "Security scanning completed!"
echo "Reports saved to: ${REPORTS_DIR}"
echo "==========================================${NC}"

echo -e "\n${YELLOW}IMPORTANT: These scans are preliminary checks only.${NC}"
echo "Professional security testing by SAP-approved vendors is required before production deployment."
echo ""
echo "Next steps:"
echo "1. Review all scan reports in ${REPORTS_DIR}"
echo "2. Address any findings before scheduling professional audit"
echo "3. Contact SAP security team to schedule official security assessment"