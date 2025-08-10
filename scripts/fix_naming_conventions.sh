#!/bin/bash
# fix-naming-conventions.sh
# Script to standardize naming conventions according to SAP enterprise standards
# WARNING: This will rename many files and directories. Create a backup first!

set -e

echo "🔧 SAP Enterprise Naming Convention Fix Script"
echo "=============================================="
echo ""
echo "⚠️  WARNING: This script will rename many files and directories!"
echo "⚠️  Make sure you have committed all changes and created a backup."
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Operation cancelled."
    exit 1
fi

# Create a backup tag
echo "📦 Creating git tag for backup..."
git tag -a "pre-naming-convention-fix-$(date +%Y%m%d-%H%M%S)" -m "Backup before naming convention fixes"

echo ""
echo "🔄 Starting naming convention fixes..."
echo ""

# Function to safely rename with git mv
safe_rename() {
    local old_name="$1"
    local new_name="$2"
    
    if [ -e "$old_name" ]; then
        if [ "$old_name" != "$new_name" ]; then
            echo "  📝 Renaming: $old_name → $new_name"
            git mv "$old_name" "$new_name" 2>/dev/null || mv "$old_name" "$new_name"
        fi
    else
        echo "  ⚠️  Not found: $old_name"
    fi
}

# 1. Root level files
echo "1️⃣  Fixing root level files..."
safe_rename "A2A_COMPLIANCE_BRIDGING_PAPER.md" "a2a-compliance-bridging-paper.md"
safe_rename "A2A_EXECUTIVE_OVERVIEW.md" "a2a-executive-overview.md"
safe_rename "GITHUB_SECRETS_SETUP.md" "github-secrets-setup.md"
safe_rename "LOCAL_TESTING_SETUP.md" "local-testing-setup.md"

# 2. Main directories
echo ""
echo "2️⃣  Fixing main directory names..."
safe_rename "a2a_agents" "a2a-agents"
safe_rename "a2a_network" "a2a-network"

# 3. Fix subdirectories in a2a-agents
echo ""
echo "3️⃣  Fixing subdirectories in a2a-agents..."
if [ -d "a2a-agents" ]; then
    cd a2a-agents
    
    # Backend subdirectories
    if [ -d "backend/app" ]; then
        cd backend/app
        safe_rename "a2a_registry" "a2a-registry"
        safe_rename "a2a_trustsystem" "a2a-trust-system"
        safe_rename "ord_registry" "ord-registry"
        safe_rename "blockchain_contracts" "blockchain-contracts"
        cd ../..
    fi
    
    # Fix report files
    echo ""
    echo "4️⃣  Fixing report files in a2a-agents..."
    safe_rename "CODE_QUALITY_IMPROVEMENT_REPORT.md" "code-quality-improvement-report.md"
    safe_rename "EMERGENCY_DEDUPLICATION_REPORT.md" "emergency-deduplication-report.md"
    
    if [ -d "backend" ]; then
        cd backend
        safe_rename "ASYNC_AWAIT_STANDARDIZATION_REPORT.md" "async-await-standardization-report.md"
        safe_rename "CONTEXTUAL_HELP_SYSTEM_IMPLEMENTATION_REPORT.md" "contextual-help-system-implementation-report.md"
        safe_rename "FINAL_POLISH_REPORT.md" "final-polish-report.md"
        safe_rename "HELP_SYSTEM_VERIFICATION_REPORT.md" "help-system-verification-report.md"
        safe_rename "INTEGRATION_STATUS_REPORT.md" "integration-status-report.md"
        safe_rename "IN_APP_HELP_INTEGRATION_ANALYSIS.md" "in-app-help-integration-analysis.md"
        safe_rename "JSDOC_DOCUMENTATION_IMPLEMENTATION_REPORT.md" "jsdoc-documentation-implementation-report.md"
        safe_rename "LAUNCH_PAD_ACCESSIBILITY_ANALYSIS.md" "launch-pad-accessibility-analysis.md"
        safe_rename "MISSING_SCREENS_IMPLEMENTATION_REPORT.md" "missing-screens-implementation-report.md"
        safe_rename "NAVIGATION_FIX_REPORT.md" "navigation-fix-report.md"
        safe_rename "QUALITY_UPGRADE_COMPLETION_REPORT.md" "quality-upgrade-completion-report.md"
        safe_rename "SAP_AUTHENTICITY_ASSESSMENT.md" "sap-authenticity-assessment.md"
        safe_rename "SAP_COMPLIANCE_100_PERCENT_REPORT.md" "sap-compliance-100-percent-report.md"
        safe_rename "SAP_ENTERPRISE_TRANSFORMATION_REPORT.md" "sap-enterprise-transformation-report.md"
        safe_rename "SCREEN_QUALITY_COMPARISON_ANALYSIS.md" "screen-quality-comparison-analysis.md"
        safe_rename "STANDARDIZED_LOGGING_IMPLEMENTATION_REPORT.md" "standardized-logging-implementation-report.md"
        safe_rename "SUPABASE_TO_SQLITE_MIGRATION_SUMMARY.md" "supabase-to-sqlite-migration-summary.md"
        safe_rename "UI_SCREENS_CATALOG.md" "ui-screens-catalog.md"
        safe_rename "UTILITY_DOCUMENTATION.md" "utility-documentation.md"
        safe_rename "VERIFICATION_AUDIT_REPORT.md" "verification-audit-report.md"
        safe_rename "PRODUCTION_DEPLOYMENT.md" "production-deployment.md"
        
        # Service documentation
        if [ -d "services" ]; then
            cd services
            safe_rename "A2A_DEPLOYMENT_GUIDE.md" "a2a-deployment-guide.md"
            safe_rename "A2A_MICROSERVICE_ARCHITECTURE.md" "a2a-microservice-architecture.md"
            cd ..
        fi
        
        # Docs folder
        if [ -d "docs" ]; then
            cd docs
            safe_rename "API_DOCUMENTATION_ENHANCED.md" "api-documentation-enhanced.md"
            safe_rename "CONTEXTUAL_HELP_IMPLEMENTATION.md" "contextual-help-implementation.md"
            safe_rename "INTEGRATION_AND_DATA_FLOW.md" "integration-and-data-flow.md"
            safe_rename "JSDOC_STYLE_GUIDE.md" "jsdoc-style-guide.md"
            safe_rename "SAP_CAP_SERVICE_DOCUMENTATION.md" "sap-cap-service-documentation.md"
            safe_rename "SAP_COMPLIANCE_GUIDE.md" "sap-compliance-guide.md"
            safe_rename "SECURITY_AUDIT_FRAMEWORK.md" "security-audit-framework.md"
            safe_rename "SECURITY_IMPLEMENTATION_GUIDE.md" "security-implementation-guide.md"
            cd ..
        fi
        
        cd ..
    fi
    
    cd ..
fi

# 5. Fix a2a-network files
echo ""
echo "5️⃣  Fixing files in a2a-network..."
if [ -d "a2a-network" ]; then
    cd a2a-network
    safe_rename "CHANGELOG.md" "changelog.md"
    safe_rename "DEPLOYMENT_SUMMARY.md" "deployment-summary.md"
    safe_rename "FIXES-COMPLETED.md" "fixes-completed.md"
    safe_rename "PRODUCTION_READINESS.md" "production-readiness.md"
    safe_rename "SECURITY.md" "security.md"
    safe_rename "README_CAP.md" "readme-cap.md"
    
    # Python SDK directory
    safe_rename "python_sdk" "python-sdk"
    
    # Documentation files
    if [ -d "docs" ]; then
        cd docs
        safe_rename "ADMIN_GUIDE.md" "admin-guide.md"
        safe_rename "API_REFERENCE.md" "api-reference.md"
        safe_rename "I18N_GUIDE.md" "i18n-guide.md"
        safe_rename "INSTALLATION.md" "installation.md"
        safe_rename "MONITORING.md" "monitoring.md"
        safe_rename "SAP-UI5-COMPLIANCE.md" "sap-ui5-compliance.md"
        safe_rename "SECURITY_AUDIT_FRAMEWORK.md" "security-audit-framework.md"
        safe_rename "SECURITY_COMPLIANCE_CHECKLIST.md" "security-compliance-checklist.md"
        safe_rename "SECURITY_TESTING_PROCEDURES.md" "security-testing-procedures.md"
        safe_rename "USER_GUIDE.md" "user-guide.md"
        safe_rename "API_REFERENCE_ENTERPRISE.md" "api-reference-enterprise.md"
        cd ..
    fi
    
    cd ..
fi

echo ""
echo "6️⃣  Updating references in code files..."

# Update references in all text files
echo "  📝 Updating directory references..."
find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" -o -name "*.cds" -o -name "*.xml" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./out/*" \
    -not -path "./dist/*" \
    -exec sed -i.bak 's/a2a_agents/a2a-agents/g' {} \;

find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" -o -name "*.cds" -o -name "*.xml" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -not -path "./out/*" \
    -not -path "./dist/*" \
    -exec sed -i.bak 's/a2a_network/a2a-network/g' {} \;

# Update specific subdirectory references
find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -exec sed -i.bak 's/a2a_registry/a2a-registry/g' {} \;

find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -exec sed -i.bak 's/a2a_trustsystem/a2a-trust-system/g' {} \;

find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -exec sed -i.bak 's/ord_registry/ord-registry/g' {} \;

find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -exec sed -i.bak 's/blockchain_contracts/blockchain-contracts/g' {} \;

find . -type f \( -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" \) \
    -not -path "./node_modules/*" \
    -not -path "./.git/*" \
    -exec sed -i.bak 's/python_sdk/python-sdk/g' {} \;

# Clean up backup files
echo "  🧹 Cleaning up backup files..."
find . -name "*.bak" -type f -delete

echo ""
echo "✅ Naming convention fixes completed!"
echo ""
echo "📋 Next steps:"
echo "  1. Review the changes with: git status"
echo "  2. Test the application to ensure nothing is broken"
echo "  3. Update any CI/CD configurations if needed"
echo "  4. Commit the changes: git commit -m 'fix: standardize naming conventions to SAP standards'"
echo ""
echo "💡 If something went wrong, you can restore from the git tag created at the beginning."

# Create a summary report
echo ""
echo "📊 Creating summary report..."
cat > naming-convention-changes.log << EOF
Naming Convention Fix Summary
Generated: $(date)

Major Changes:
- Renamed a2a_agents/ to a2a-agents/
- Renamed a2a_network/ to a2a-network/
- Converted all UPPERCASE report files to lowercase-hyphenated
- Updated all file references in code

Files Renamed: $(git status --porcelain | grep -c "^R")
Files Modified: $(git status --porcelain | grep -c "^ M")

Review full changes with: git status --porcelain
EOF

echo "📄 Summary saved to: naming-convention-changes.log"