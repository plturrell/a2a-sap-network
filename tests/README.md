# A2A Enterprise Test Suite

This directory follows SAP enterprise standards for test organization and execution.

## Test Structure

```
tests/
├── unit/                    # Fast, isolated unit tests
│   ├── a2aAgents/
│   │   ├── agents/         # Agent-specific unit tests (13 files)
│   │   ├── services/       # Service unit tests (30 files)
│   │   └── components/     # Component unit tests
│   └── a2aNetwork/
├── integration/             # Tests requiring external services
│   ├── a2aAgents/
│   │   ├── workflows/      # Workflow integration tests (5 files)
│   │   ├── blockchain/     # Blockchain tests (9 files)
│   │   ├── network/        # Network integration tests
│   │   └── registry/       # Registry tests (3 files)
│   └── a2aNetwork/
│       └── registry/       # Network registry tests (3 files)
├── e2e/                     # End-to-end UI tests (Cypress)
│   ├── a2aAgents/          # Agent portal UI tests
│   └── a2aNetwork/         # Network app UI tests
├── performance/             # Performance and load tests
├── security/                # Security tests
├── accessibility/           # WCAG compliance tests
├── localization/           # i18n tests
├── mobile/                 # Mobile responsiveness tests
└── browserCompat/          # Cross-browser tests
```

## Running Tests

### All Tests
```bash
pytest
```

### By Category
```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Security tests
pytest -m security

# Specific component
pytest -m agent
pytest -m blockchain
```

### By Directory
```bash
# All agent unit tests
pytest tests/unit/a2aAgents/

# All blockchain tests
pytest tests/integration/a2aAgents/blockchain/

# All E2E tests
npm run cypress:run
```

### Quick Mode
```bash
# Skip slow tests
pytest --quick
```

### Coverage Report
```bash
# Generate HTML coverage report
pytest --cov --cov-report=html

# View coverage
open htmlcov/index.html
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Tests taking > 5 seconds
- `@pytest.mark.network` - Tests requiring network
- `@pytest.mark.blockchain` - Blockchain tests
- `@pytest.mark.agent` - Agent-specific tests

## Configuration Files

- `pytest.ini` - Main pytest configuration
- `conftest.py` - Shared fixtures and test setup
- `cypress.config.js` - Cypress E2E configuration

## Test Data

Test fixtures and mock data are located in:
- Unit test fixtures: Within each test file
- Integration test data: `tests/fixtures/`
- E2E test data: `cypress/fixtures/`

## Writing New Tests

1. Place tests in the appropriate directory based on type
2. Use descriptive test names: `test_<feature>_<scenario>`
3. Include docstrings explaining what is being tested
4. Use appropriate markers for categorization
5. Mock external dependencies in unit tests
6. Follow the existing test patterns

## CI/CD Integration

Tests are automatically run in CI/CD pipeline:
- Unit tests: On every commit
- Integration tests: On PR creation
- E2E tests: Before deployment
- Performance tests: Weekly schedule

## Consolidation Summary

Successfully consolidated 108 test files:
- Moved 72 Python test files from scattered locations
- Organized 17 Solidity test files (already in correct location)
- Structured 9 JavaScript test files
- Created proper test categorization
- Removed redundant test files
- Updated all import paths and configurations

The test suite is now properly organized and ready for efficient test execution and maintenance.