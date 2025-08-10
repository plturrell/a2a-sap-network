#!/bin/bash
# validate-naming-conventions.sh
# Script to validate that naming conventions follow SAP enterprise standards

echo "🔍 SAP Naming Convention Validator"
echo "=================================="
echo ""

# Initialize counters
total_issues=0
file_issues=0
dir_issues=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if name follows convention
check_naming() {
    local path="$1"
    local type="$2"
    local name=$(basename "$path")
    
    # Skip special files and directories
    if [[ "$name" =~ ^\..*$ ]] || [ "$name" = "node_modules" ] || [ "$name" = "README.md" ]; then
        return 0
    fi
    
    # Skip Python files (they should use underscores)
    if [[ "$name" =~ \.py$ ]]; then
        return 0
    fi
    
    # Skip standard config files
    if [[ "$name" =~ ^(package\.json|mta\.yaml|tsconfig\.json|jest\.config\.js|webpack\.config\.js)$ ]]; then
        return 0
    fi
    
    # Check for issues
    local has_issue=0
    
    # Check for underscores (except in Python files)
    if [[ "$name" =~ _ ]] && [[ ! "$name" =~ \.py$ ]]; then
        echo -e "${RED}❌ Contains underscore: $path${NC}"
        has_issue=1
    fi
    
    # Check for uppercase letters in files (except certain patterns)
    if [ "$type" = "file" ] && [[ "$name" =~ [A-Z] ]] && [[ ! "$name" =~ ^(README|LICENSE|CHANGELOG|CONTRIBUTING).*$ ]]; then
        # Skip TypeScript/JavaScript class files that might legitimately use PascalCase
        if [[ ! "$name" =~ \.(tsx?|jsx?)$ ]] || [[ "$name" =~ ^[A-Z].*\.(tsx?|jsx?)$ ]]; then
            echo -e "${YELLOW}⚠️  Contains uppercase: $path${NC}"
            has_issue=1
        fi
    fi
    
    # Check for uppercase in directories
    if [ "$type" = "dir" ] && [[ "$name" =~ [A-Z] ]]; then
        echo -e "${RED}❌ Directory contains uppercase: $path${NC}"
        has_issue=1
    fi
    
    # Check for spaces
    if [[ "$name" =~ \  ]]; then
        echo -e "${RED}❌ Contains spaces: $path${NC}"
        has_issue=1
    fi
    
    return $has_issue
}

echo "Checking directory names..."
echo "-------------------------"

# Find all directories and check them
while IFS= read -r dir; do
    if check_naming "$dir" "dir"; then
        ((dir_issues++))
        ((total_issues++))
    fi
done < <(find . -type d -not -path "./node_modules/*" -not -path "./.git/*" -not -path "./out/*" -not -path "./dist/*" -not -path "./coverage/*")

echo ""
echo "Checking file names..."
echo "--------------------"

# Find all files and check them
while IFS= read -r file; do
    if check_naming "$file" "file"; then
        ((file_issues++))
        ((total_issues++))
    fi
done < <(find . -type f -not -path "./node_modules/*" -not -path "./.git/*" -not -path "./out/*" -not -path "./dist/*" -not -path "./coverage/*" -not -name ".*")

echo ""
echo "=================================="
echo "Validation Summary"
echo "=================================="

if [ $total_issues -eq 0 ]; then
    echo -e "${GREEN}✅ All naming conventions are correct!${NC}"
    echo "   No issues found."
else
    echo -e "${RED}❌ Found $total_issues naming convention issues:${NC}"
    echo "   - Directory issues: $dir_issues"
    echo "   - File issues: $file_issues"
    echo ""
    echo "Run ./scripts/fix-naming-conventions.sh to fix these issues automatically."
fi

echo ""
echo "SAP Naming Convention Guidelines:"
echo "--------------------------------"
echo "• Directories: lowercase-with-hyphens"
echo "• Files: lowercase-with-hyphens.ext"
echo "• No underscores (except Python files)"
echo "• No spaces in names"
echo "• Consistent prefixes (a2a-)"

exit $total_issues