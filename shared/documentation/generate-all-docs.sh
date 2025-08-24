#!/bin/bash
# Comprehensive Documentation Generation Script for A2A Platform
# This script generates all types of documentation for the A2A platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ROOT_DIR="${1:-$(pwd)}"
OUTPUT_DIR="${2:-${ROOT_DIR}/docs}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SKIP_INSTALL="${SKIP_INSTALL:-false}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install dependencies
install_dependencies() {
    if [[ "$SKIP_INSTALL" == "true" ]]; then
        log_info "Skipping dependency installation"
        return
    fi

    log_info "Installing documentation generation dependencies..."
    
    # Node.js dependencies
    if command_exists npm; then
        cd "$(dirname "$0")"
        npm install --silent
        cd - > /dev/null
        log_success "Node.js dependencies installed"
    else
        log_warning "npm not found, skipping Node.js dependency installation"
    fi
    
    # Python dependencies
    if command_exists python3; then
        python3 -m pip install --quiet --user ast-comments docstring-parser pathlib2 2>/dev/null || true
        log_success "Python dependencies installed"
    else
        log_warning "python3 not found, skipping Python dependency installation"
    fi
}

# Function to create output directories
create_output_dirs() {
    log_info "Creating output directories..."
    mkdir -p "${OUTPUT_DIR}/generated"
    mkdir -p "${OUTPUT_DIR}/python"
    mkdir -p "${OUTPUT_DIR}/api"
    mkdir -p "${OUTPUT_DIR}/schemas"
    mkdir -p "${OUTPUT_DIR}/agents"
    mkdir -p "${OUTPUT_DIR}/configs"
    log_success "Output directories created"
}

# Function to generate JavaScript/TypeScript documentation
generate_js_docs() {
    log_info "Generating JavaScript/TypeScript documentation..."
    
    local doc_generator="$(dirname "$0")/doc-generator.js"
    
    if [[ -f "$doc_generator" ]]; then
        if command_exists node; then
            node "$doc_generator" "$ROOT_DIR" "${OUTPUT_DIR}/generated" 2>/dev/null || {
                log_warning "JavaScript documentation generation encountered issues"
                return 1
            }
            log_success "JavaScript/TypeScript documentation generated"
        else
            log_error "Node.js not found, cannot generate JS documentation"
            return 1
        fi
    else
        log_error "doc-generator.js not found at $doc_generator"
        return 1
    fi
}

# Function to generate Python documentation
generate_python_docs() {
    log_info "Generating Python documentation..."
    
    local python_generator="$(dirname "$0")/generate-docs.py"
    
    if [[ -f "$python_generator" ]]; then
        if command_exists python3; then
            python3 "$python_generator" "$ROOT_DIR" --output "${OUTPUT_DIR}/python" --log-level "$LOG_LEVEL" || {
                log_warning "Python documentation generation encountered issues"
                return 1
            }
            log_success "Python documentation generated"
        else
            log_error "Python 3 not found, cannot generate Python documentation"
            return 1
        fi
    else
        log_error "generate-docs.py not found at $python_generator"
        return 1
    fi
}

# Function to generate OpenAPI documentation
generate_openapi_docs() {
    log_info "Generating OpenAPI documentation..."
    
    # Find OpenAPI/Swagger files
    local openapi_files=($(find "$ROOT_DIR" -name "*.yaml" -o -name "*.yml" -o -name "openapi.*" | grep -E "(openapi|swagger)" | head -5))
    
    if [[ ${#openapi_files[@]} -eq 0 ]]; then
        log_warning "No OpenAPI files found"
        return 0
    fi
    
    # Generate documentation for each OpenAPI file
    for file in "${openapi_files[@]}"; do
        local basename=$(basename "$file" .yaml)
        basename=$(basename "$basename" .yml)
        
        log_info "Processing OpenAPI file: $file"
        
        # Create a simple markdown documentation
        cat > "${OUTPUT_DIR}/api/${basename}-api.md" << EOF
# ${basename} API Documentation

**Source file:** \`$(realpath --relative-to="$ROOT_DIR" "$file")\`

## Overview

This API documentation was generated from the OpenAPI specification file.

\`\`\`yaml
$(head -50 "$file")
...
\`\`\`

**Full specification:** [$(basename "$file")]($(realpath --relative-to="$OUTPUT_DIR/api" "$file"))

## Generation Info
- Generated on: $(date)
- Generator: A2A Documentation System
EOF

    done
    
    log_success "OpenAPI documentation generated for ${#openapi_files[@]} files"
}

# Function to generate agent documentation
generate_agent_docs() {
    log_info "Generating agent-specific documentation..."
    
    # Find agent directories
    local agent_dirs=($(find "$ROOT_DIR" -type d -name "*agent*" -o -name "*Agent*" | head -10))
    
    if [[ ${#agent_dirs[@]} -eq 0 ]]; then
        log_warning "No agent directories found"
        return 0
    fi
    
    # Generate documentation for each agent
    for dir in "${agent_dirs[@]}"; do
        local agent_name=$(basename "$dir")
        
        log_info "Documenting agent: $agent_name"
        
        # Create agent documentation
        cat > "${OUTPUT_DIR}/agents/${agent_name}.md" << EOF
# ${agent_name} Documentation

**Directory:** \`$(realpath --relative-to="$ROOT_DIR" "$dir")\`

## Overview

Auto-generated documentation for ${agent_name}.

## Files

EOF
        
        # List key files in agent directory
        find "$dir" -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" \) | head -10 | while read -r file; do
            echo "- [\`$(basename "$file")\`]($(realpath --relative-to="$OUTPUT_DIR/agents" "$file"))" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
        done
        
        cat >> "${OUTPUT_DIR}/agents/${agent_name}.md" << EOF

## Configuration

EOF
        
        # Look for configuration files
        find "$dir" -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "config.*" | head -5 | while read -r config_file; do
            echo "### $(basename "$config_file")" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
            echo "\`\`\`" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
            head -20 "$config_file" >> "${OUTPUT_DIR}/agents/${agent_name}.md" 2>/dev/null || echo "Unable to read file" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
            echo "\`\`\`" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
            echo "" >> "${OUTPUT_DIR}/agents/${agent_name}.md"
        done
        
    done
    
    log_success "Agent documentation generated for ${#agent_dirs[@]} agents"
}

# Function to generate configuration documentation
generate_config_docs() {
    log_info "Generating configuration documentation..."
    
    # Find configuration files
    local config_files=($(find "$ROOT_DIR" -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "config.*" -o -name "*.config.*" | grep -v node_modules | head -15))
    
    if [[ ${#config_files[@]} -eq 0 ]]; then
        log_warning "No configuration files found"
        return 0
    fi
    
    # Generate master configuration documentation
    cat > "${OUTPUT_DIR}/configs/README.md" << EOF
# Configuration Documentation

This document provides an overview of all configuration files in the A2A platform.

**Generated on:** $(date)

## Configuration Files

EOF
    
    # Document each configuration file
    for file in "${config_files[@]}"; do
        local basename=$(basename "$file")
        local dirname=$(dirname "$file")
        local relative_path=$(realpath --relative-to="$ROOT_DIR" "$file")
        
        echo "### $basename" >> "${OUTPUT_DIR}/configs/README.md"
        echo "**Path:** \`$relative_path\`" >> "${OUTPUT_DIR}/configs/README.md"
        echo "" >> "${OUTPUT_DIR}/configs/README.md"
        
        # Show file preview
        echo "\`\`\`yaml" >> "${OUTPUT_DIR}/configs/README.md"
        head -15 "$file" >> "${OUTPUT_DIR}/configs/README.md" 2>/dev/null || echo "Unable to read file"
        echo "\`\`\`" >> "${OUTPUT_DIR}/configs/README.md"
        echo "" >> "${OUTPUT_DIR}/configs/README.md"
        
    done
    
    log_success "Configuration documentation generated for ${#config_files[@]} files"
}

# Function to generate master index
generate_master_index() {
    log_info "Generating master documentation index..."
    
    cat > "${OUTPUT_DIR}/README.md" << EOF
# A2A Platform Documentation

> Comprehensive auto-generated documentation for the A2A Platform

**Generated on:** $(date)  
**Root directory:** \`$ROOT_DIR\`

## 📚 Documentation Sections

### 🌐 General Documentation
- [Generated Documentation](./generated/README.md) - Auto-generated from code comments and schemas
- [API Reference](./api/) - OpenAPI and service documentation  

### 🐍 Python Documentation
- [Python API Reference](./python/README.md) - Complete Python codebase documentation
- [Python Modules](./python/modules/) - Individual module documentation
- [Class Hierarchy](./python/class-hierarchy.md) - Class inheritance structure

### 🤖 Agent Documentation  
- [Agent Overview](./agents/) - Documentation for all agents in the platform

### ⚙️ Configuration Documentation
- [Configuration Files](./configs/README.md) - All configuration files and settings

## 📊 Generation Statistics

EOF
    
    # Add statistics
    echo "- **Total documentation files generated:** $(find "$OUTPUT_DIR" -name "*.md" | wc -l)" >> "${OUTPUT_DIR}/README.md"
    echo "- **JavaScript/TypeScript files processed:** $(find "$ROOT_DIR" -name "*.js" -o -name "*.ts" | wc -l)" >> "${OUTPUT_DIR}/README.md"
    echo "- **Python files processed:** $(find "$ROOT_DIR" -name "*.py" | wc -l)" >> "${OUTPUT_DIR}/README.md"
    echo "- **Configuration files found:** $(find "$ROOT_DIR" -name "*.json" -o -name "*.yaml" -o -name "*.yml" | grep -v node_modules | wc -l)" >> "${OUTPUT_DIR}/README.md"
    
    cat >> "${OUTPUT_DIR}/README.md" << EOF

## 🔧 Regeneration

To regenerate this documentation, run:

\`\`\`bash
./shared/documentation/generate-all-docs.sh $ROOT_DIR $OUTPUT_DIR
\`\`\`

## 📝 Notes

- This documentation is auto-generated and should not be edited manually
- To update documentation, modify the source code comments and regenerate
- For questions about the documentation system, see the shared/documentation/ directory

---

*Generated by A2A Platform Documentation System*
EOF
    
    log_success "Master documentation index generated"
}

# Main execution
main() {
    log_info "Starting A2A Platform documentation generation"
    log_info "Root directory: $ROOT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    
    # Check if root directory exists
    if [[ ! -d "$ROOT_DIR" ]]; then
        log_error "Root directory does not exist: $ROOT_DIR"
        exit 1
    fi
    
    # Create output directories
    create_output_dirs
    
    # Install dependencies
    install_dependencies
    
    # Generate different types of documentation
    local success_count=0
    local total_count=0
    
    # JavaScript/TypeScript documentation
    ((total_count++))
    if generate_js_docs; then
        ((success_count++))
    fi
    
    # Python documentation
    ((total_count++))
    if generate_python_docs; then
        ((success_count++))
    fi
    
    # OpenAPI documentation
    ((total_count++))
    if generate_openapi_docs; then
        ((success_count++))
    fi
    
    # Agent documentation
    ((total_count++))
    if generate_agent_docs; then
        ((success_count++))
    fi
    
    # Configuration documentation
    ((total_count++))
    if generate_config_docs; then
        ((success_count++))
    fi
    
    # Generate master index
    generate_master_index
    
    # Final summary
    echo ""
    log_success "Documentation generation completed!"
    log_info "Success rate: $success_count/$total_count generators"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Total documentation files: $(find "$OUTPUT_DIR" -name "*.md" | wc -l)"
    
    if [[ $success_count -eq $total_count ]]; then
        log_success "All documentation generators completed successfully"
        exit 0
    else
        log_warning "Some documentation generators encountered issues"
        exit 1
    fi
}

# Run main function
main "$@"