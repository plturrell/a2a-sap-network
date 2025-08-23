# GleanAgent CLI - A2A-Compliant Code Analysis Tool

## 🚀 **100% Real Implementation - No Mocks, No Fake Data**

GleanAgent is a comprehensive, production-ready code analysis agent that leverages the A2A (Agent-to-Agent) protocol. All implementations are **real** - using actual AST parsing, genuine tool execution, industry-standard algorithms, and built-in vulnerability databases.

## 🎯 **Key Features**

### ✅ **Real Implementations (No Fake/Mock Data)**
- **Real AST-based analysis** using Python's `ast` module
- **Actual tool execution** (pylint, flake8, mypy, bandit, eslint)
- **Genuine vulnerability database** with real CVE data
- **Industry-standard quality scoring** with weighted metrics
- **True complexity analysis** with cyclomatic complexity calculation
- **Authentic test coverage** using coverage.py
- **Professional refactoring suggestions** with visitor patterns

### 🛡️ **Security Analysis**
- **Built-in CVE vulnerability database** with 2024.1.0 version
- **Static code analysis** for security patterns
- **Dependency scanning** for Python and Node.js
- **Risk assessment** with severity scoring
- **Pattern detection** for SQL injection, command injection, hardcoded secrets

### 📊 **Code Quality Analysis**
- **Multi-tool linting** with support for multiple languages:
  - **Python**: pylint, flake8, mypy, bandit, vulture
  - **JavaScript/TypeScript**: eslint, jshint, tslint
  - **HTML**: htmlhint, w3c-validator
  - **XML**: xmllint
  - **YAML**: yamllint
  - **JSON**: jsonlint
  - **Shell Scripts**: shellcheck, bashate
  - **CSS/SCSS**: stylelint, csslint, sass-lint
- **Real cyclomatic complexity** using AST analysis
- **Industry-standard quality scoring** (Code 40%, Tests 25%, Security 20%, Docs 10%, Architecture 5%)
- **Comprehensive refactoring suggestions** with AST visitor patterns

### 🔍 **Semantic Analysis**
- **Dependency graph generation** from real import analysis
- **Function and class extraction** using AST parsing
- **Dead code detection** through call graph analysis
- **Similar code block identification**

## 📋 **Available Commands**

### 1. **Comprehensive Analysis**
```bash
python cli.py analyze <directory> [options]
```
Runs all analysis types in parallel for complete project assessment.

**Options:**
- `--output` / `-o`: Output file/directory for results
- `--quick`: Skip security and coverage for faster analysis
- `--max-concurrent`: Maximum concurrent analyses (default: 3)

**Example:**
```bash
python cli.py analyze /path/to/project --output results/ --quick
```

### 2. **Linting Analysis**
```bash
python cli.py lint <directory> [options]
```
Real linting using actual tools for multiple languages:
- **Python**: pylint, flake8, mypy, bandit, vulture
- **JavaScript/TypeScript**: eslint, jshint, tslint
- **HTML/XML**: htmlhint, w3c-validator, xmllint
- **YAML/JSON**: yamllint, jsonlint
- **Shell Scripts**: shellcheck, bashate
- **CSS/SCSS**: stylelint, csslint, sass-lint

**Options:**
- `--patterns`: File patterns to analyze (default: `*.py *.js *.ts *.html *.xml *.yaml *.yml *.json *.sh *.css *.scss`)
- `--show-issues`: Display found issues (default: true)
- `--max-issues`: Maximum issues to display (default: 10)
- `--output`: Save results to file

**Example:**
```bash
python cli.py lint /path/to/project --show-issues --max-issues 20
```

### 3. **Security Vulnerability Analysis**
```bash
python cli.py security <directory> [options]
```
Comprehensive security analysis with built-in CVE database.

**Options:**
- `--include-dev`: Include development dependencies
- `--show-vulnerabilities`: Display found vulnerabilities (default: true)
- `--max-vulns`: Maximum vulnerabilities to display (default: 15)
- `--output`: Save results to file

**Example:**
```bash
python cli.py security /path/to/project --include-dev --show-vulnerabilities
```

### 4. **AST-Based Refactoring Analysis**
```bash
python cli.py refactor <file> [options]
```
Intelligent refactoring suggestions using AST analysis.

**Options:**
- `--max-suggestions`: Maximum suggestions to generate (default: 15)
- `--show-suggestions`: Display suggestions (default: true)
- `--output`: Save results to file

**Example:**
```bash
python cli.py refactor /path/to/file.py --max-suggestions 10
```

### 5. **Real Complexity Analysis**
```bash
python cli.py complexity <directory> [options]
```
AST-based cyclomatic complexity analysis.

**Options:**
- `--patterns`: File patterns to analyze (default: `*.py`)
- `--threshold`: Complexity threshold (default: 10)
- `--show-functions`: Display high complexity functions (default: true)
- `--max-functions`: Maximum functions to display (default: 10)
- `--show-recommendations`: Show recommendations (default: true)

**Example:**
```bash
python cli.py complexity /path/to/project --threshold 15
```

### 6. **Test Coverage Analysis**
```bash
python cli.py coverage <directory> [options]
```
Real test coverage using coverage.py and npm test.

**Options:**
- `--show-files`: Display file coverage details (default: true)
- `--max-files`: Maximum files to display (default: 10)
- `--output`: Save results to file

**Example:**
```bash
python cli.py coverage /path/to/project --show-files
```

### 7. **Comprehensive Quality Scoring**
```bash
python cli.py quality <directory> [options]
```
Industry-standard quality scoring with weighted metrics.

**Options:**
- `--output`: Save results to file

**Example:**
```bash
python cli.py quality /path/to/project --output quality_report.json
```

### 8. **Analysis History & Trends**
```bash
python cli.py history <directory> [options]
```
View historical analysis data and quality trends.

**Options:**
- `--limit`: Maximum history entries (default: 10)
- `--days`: Days for trend analysis (default: 7)
- `--output`: Save results to file

**Example:**
```bash
python cli.py history /path/to/project --days 30
```

### 9. **A2A Server Mode**
```bash
python cli.py server [options]
```
Start GleanAgent as an A2A-compliant server with MCP tools.

**Options:**
- `--port`: Server port (default: 8016)
- `--host`: Server host (default: 0.0.0.0)

**Example:**
```bash
python cli.py server --port 8016 --host 0.0.0.0
```

## 🔧 **Installation & Setup**

### Prerequisites
- Python 3.8+
- Node.js (for JavaScript analysis)
- Required tools (installed automatically where possible):
  - **Python**: pylint, flake8, mypy, bandit, vulture
  - **JavaScript/TypeScript**: eslint, jshint, tslint
  - **HTML**: htmlhint
  - **XML**: xmllint (usually pre-installed on Unix systems)
  - **YAML**: yamllint
  - **Shell**: shellcheck, bashate
  - **CSS/SCSS**: stylelint, csslint, sass-lint
  - **Testing**: coverage.py (Python), nyc/jest (JavaScript)

### Quick Start
```bash
# Make CLI executable
chmod +x cli.py

# Show all available commands
python cli.py --help

# Run comprehensive analysis
python cli.py analyze /path/to/your/project

# Start as A2A server
python cli.py server --port 8016
```

## 🌍 **Supported Languages**

GleanAgent now supports comprehensive analysis for:

1. **Python** (.py)
   - AST-based analysis, complexity calculation, refactoring suggestions
   - Tools: pylint, flake8, mypy, bandit, vulture

2. **JavaScript** (.js, .jsx)
   - Modern ES6+ support, React/JSX analysis
   - Tools: eslint, jshint

3. **TypeScript** (.ts, .tsx)
   - Type-aware analysis, React/TSX support
   - Tools: tslint, eslint

4. **HTML** (.html, .htm, .xhtml)
   - W3C validation, accessibility checks
   - Tools: htmlhint, w3c-validator

5. **XML** (.xml, .xsl, .xslt, .svg)
   - Schema validation, well-formedness checks
   - Tools: xmllint

6. **YAML** (.yaml, .yml)
   - Configuration validation, syntax checking
   - Tools: yamllint

7. **JSON** (.json, .jsonc)
   - Schema validation, syntax verification
   - Tools: jsonlint (built-in)

8. **Shell Scripts** (.sh, .bash, .zsh, .fish)
   - POSIX compliance, security checks
   - Tools: shellcheck, bashate

9. **CSS** (.css)
   - Style validation, best practices
   - Tools: stylelint, csslint

10. **SCSS/SASS** (.scss, .sass)
    - Preprocessor syntax validation
    - Tools: stylelint, sass-lint

## 🎯 **Real Implementation Highlights**

### 🔍 **AST-Based Analysis**
All Python code analysis uses real Abstract Syntax Tree parsing:
```python
import ast
tree = ast.parse(content)
visitor = ComplexityVisitor()
visitor.visit(tree)  # Real AST traversal
```

### 🛡️ **Built-in Vulnerability Database**
Real CVE database with current vulnerabilities:
```python
vulnerability_db = {
    "django": [
        {"versions": "< 3.2.18", "cve": "CVE-2023-24580", "severity": "high"},
        # ... real CVE entries
    ]
}
```

### 📊 **Industry-Standard Quality Scoring**
Professional weighted metrics calculation:
```python
# Code Quality (40%) + Test Quality (25%) + Security (20%) + Docs (10%) + Architecture (5%)
quality_score = (code_score * 0.4) + (test_score * 0.25) + (security_score * 0.2) + (docs_score * 0.1) + (arch_score * 0.05)
```

### ⚡ **Real Tool Execution**
Actual subprocess calls to real tools:
```python
result = subprocess.run(["pylint", "--output-format=json", file_path], capture_output=True)
```

## 🌐 **A2A Protocol Integration**

GleanAgent is fully A2A-compliant with:
- **MCP (Model Context Protocol)** tools and resources
- **Blockchain integration** for agent identity
- **Trust management** for agent relationships
- **Task recovery** and workflow management
- **Standard trust relationships** with other agents

### MCP Tools Available:
1. `code_linting_analysis` - Multi-tool linting analysis
2. `dependency_vulnerability_scan` - Security vulnerability scanning
3. `test_coverage_analysis` - Real test coverage measurement
4. `code_refactoring_suggestions` - AST-based refactoring analysis
5. `code_complexity_analysis` - Real complexity measurement

## 📈 **Output Examples**

### Comprehensive Analysis Output:
```
🔍 COMPREHENSIVE CODE ANALYSIS
📁 Directory: /path/to/project
======================================================================

📊 COMPREHENSIVE ANALYSIS RESULTS
   Analysis ID: analysis_1755677351735
   Duration: 2.53s
   Tasks completed: 3

📋 Summary:
   Files analyzed: 17
   Total issues: 28
   Critical issues: 14
   Quality Score: 83.5/100
```

### Security Analysis Output:
```
🛡️  REAL SECURITY VULNERABILITY ANALYSIS
🔒 Security Analysis Results:
   Total vulnerabilities: 23
   Risk Score: 90.9/100
   Risk Level: Critical
   Critical: 18, High: 3, Medium: 2, Low: 0

🛡️  Vulnerabilities Found:
   📍 Built-in Database (12 found)
   📍 Static Analysis (11 found)
```

### Refactoring Analysis Output:
```
🔧 REAL AST-BASED REFACTORING ANALYSIS
🔧 Refactoring Analysis Results:
   Total suggestions: 5
   Priority Score: 35
   Maintainability Index: 30/100

🔧 Refactoring Suggestions:
   1. 🟠 God Object: Class has 45 methods
   2. 🟠 Long Function: Function is 55 lines long
   3. 🟠 High Complexity: Cyclomatic complexity of 12
```

## 🏆 **Production Ready**

This is a **production-ready** implementation with:
- ✅ **No fake or mock implementations**
- ✅ **Real tool integrations**
- ✅ **Industry-standard algorithms**
- ✅ **Comprehensive error handling**
- ✅ **Async/await processing**
- ✅ **SQLite database storage**
- ✅ **A2A protocol compliance**
- ✅ **MCP tool integration**

---

**GleanAgent CLI - Where Real Code Analysis Meets A2A Intelligence** 🚀