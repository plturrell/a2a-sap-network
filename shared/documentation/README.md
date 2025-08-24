# A2A Platform Documentation Generator

Automated documentation generation system for the A2A Platform with support for multiple programming languages, API specifications, and configuration files.

## Features

🚀 **Multi-Language Support**
- JavaScript/TypeScript documentation with JSDoc parsing
- Python documentation with AST analysis and docstring extraction  
- OpenAPI/Swagger specification documentation
- Configuration file documentation (YAML, JSON)

📊 **Comprehensive Analysis**
- Code comments and docstrings
- Function signatures and type hints
- Class hierarchies and inheritance
- API endpoints and schemas
- Configuration parameters

🎯 **Smart Detection**
- Automatically identifies file types and content
- Extracts agent-specific documentation
- Processes modular configurations
- Generates cross-references and navigation

## Quick Start

### Generate All Documentation

```bash
# Generate documentation for current directory
./generate-all-docs.sh

# Generate for specific directory
./generate-all-docs.sh /path/to/a2a/project

# Specify custom output directory
./generate-all-docs.sh /path/to/project /path/to/docs
```

### Individual Generators

```bash
# JavaScript/TypeScript documentation
node doc-generator.js /path/to/project

# Python documentation  
python generate-docs.py /path/to/project --output ./docs/python

# Install dependencies first
cd shared/documentation && npm install
```

## Generated Documentation Structure

```
docs/
├── README.md                    # Master index
├── generated/                   # General documentation
│   ├── api-documentation.md     # API endpoints
│   ├── agent-documentation.md   # Agent details
│   ├── schema-documentation.md  # Data schemas
│   └── config-documentation.md  # Configurations
├── python/                      # Python-specific docs
│   ├── README.md               # Python overview
│   ├── api-reference.md        # Python API reference
│   ├── class-hierarchy.md      # Class inheritance
│   └── modules/                # Individual modules
├── agents/                     # Agent documentation
├── configs/                    # Configuration docs
└── api/                       # OpenAPI specs
```

## Configuration

### Environment Variables

```bash
export LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
export SKIP_INSTALL=false       # Skip dependency installation
```

### Customization Options

**JavaScript Generator (`doc-generator.js`)**
```javascript
const options = {
    rootDir: './src',
    outputDir: './docs/generated',
    includePatterns: ['**/*.js', '**/*.ts'],
    excludePatterns: ['**/node_modules/**', '**/dist/**'],
    generateApiDocs: true,
    generateAgentDocs: true,
    generateSchemaDocs: true,
    generateConfigDocs: true
};
```

**Python Generator (`generate-docs.py`)**
```bash
python generate-docs.py --help

# Options:
#   --output DIR          Output directory
#   --log-level LEVEL     Logging level (DEBUG/INFO/WARNING/ERROR)
```

## Advanced Usage

### Watch Mode (Development)

```bash
# Auto-regenerate on file changes (requires nodemon)
npm run dev
```

### Selective Generation

```bash
# Only generate Python documentation
python generate-docs.py /path/to/project

# Only generate JavaScript documentation  
node doc-generator.js /path/to/project

# Skip dependency installation
SKIP_INSTALL=true ./generate-all-docs.sh
```

### Custom Output Processing

```bash
# Generate and post-process
./generate-all-docs.sh && \
    pandoc docs/README.md -o docs/README.pdf && \
    echo "Documentation generated and converted to PDF"
```

## Integration

### CI/CD Pipeline

```yaml
# GitHub Actions example
name: Generate Documentation
on: [push, pull_request]
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Generate Documentation
        run: ./shared/documentation/generate-all-docs.sh
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
./shared/documentation/generate-all-docs.sh
git add docs/
```

## Supported Documentation Formats

### Code Comments
- JSDoc comments (`/** */`)
- Python docstrings (`"""`)
- Inline comments (`#`, `//`)

### API Specifications
- OpenAPI 3.0 (YAML/JSON)
- Swagger 2.0
- Custom API route definitions

### Configuration Files
- YAML configuration files
- JSON configuration files
- Environment files
- Docker Compose files

## Examples

### Function Documentation

```python
def analyze_codebase(directory: str, options: Dict[str, Any]) -> AnalysisResult:
    """
    Analyze a codebase directory for security vulnerabilities and code quality.
    
    Args:
        directory (str): Path to the codebase directory
        options (Dict[str, Any]): Analysis options including:
            - include_security: Enable security scanning
            - file_patterns: File patterns to analyze
            
    Returns:
        AnalysisResult: Complete analysis results with findings and metrics
        
    Raises:
        ValueError: If directory path is invalid
        SecurityError: If critical vulnerabilities are found
        
    Example:
        >>> result = analyze_codebase("/path/to/code", {"include_security": True})
        >>> print(f"Found {len(result.vulnerabilities)} vulnerabilities")
    """
```

### Class Documentation

```javascript
/**
 * Enhanced Glean Agent for comprehensive code analysis
 * 
 * @class GleanAgentEnhanced
 * @extends BaseAgent
 * 
 * @example
 * const agent = new GleanAgentEnhanced({
 *   enableSecurity: true,
 *   maxConcurrentScans: 5
 * });
 * 
 * const result = await agent.analyzeDirectory('/path/to/code');
 */
class GleanAgentEnhanced extends BaseAgent {
    /**
     * Initialize the enhanced Glean agent
     * @param {Object} config - Configuration options
     * @param {boolean} config.enableSecurity - Enable security scanning
     * @param {number} config.maxConcurrentScans - Maximum concurrent operations
     */
    constructor(config = {}) {
        // Implementation
    }
}
```

## Troubleshooting

### Common Issues

**Permission Errors**
```bash
# Make script executable
chmod +x ./shared/documentation/generate-all-docs.sh

# Fix file permissions
find ./shared/documentation -name "*.sh" -exec chmod +x {} \;
```

**Missing Dependencies**
```bash
# Install Node.js dependencies
cd shared/documentation && npm install

# Install Python dependencies
pip install ast-comments docstring-parser pathlib2
```

**Large Codebases**
```bash
# Use selective generation for large codebases
LOG_LEVEL=WARNING ./generate-all-docs.sh

# Skip non-essential directories
# Edit excludePatterns in doc-generator.js
```

### Debug Mode

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG ./generate-all-docs.sh

# Check individual generators
node doc-generator.js --help
python generate-docs.py --log-level DEBUG
```

## Contributing

### Adding New Generators

1. Create generator in appropriate language
2. Follow existing patterns for output structure
3. Add integration to `generate-all-docs.sh`
4. Update documentation

### Extending Existing Generators

1. Modify extraction patterns in source generators
2. Update output formatting functions
3. Test with various code patterns
4. Update configuration options

## License

MIT License - see project root for details.

---

**Generated by A2A Platform Documentation System**  
*For technical questions, see the A2A Platform documentation or create an issue.*