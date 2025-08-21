# A2A Network Management Makefile
.PHONY: help install test lint format clean dev-setup start-agent0 start-agent1 start-all deploy

# Default target
help:
	@echo "A2A Network Management Commands"
	@echo "==============================="
	@echo "  install       Install dependencies using pip"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  lint          Run code linting"
	@echo "  format        Format code with black and ruff"
	@echo "  clean         Clean up cache files"
	@echo "  dev-setup     Setup development environment"
	@echo ""
	@echo "Agent Management:"
	@echo "  start-registry       Start Registry Server (port 8000)"
	@echo "  start-agent0         Start Data Product Agent (port 8001)"
	@echo "  start-agent1         Start Standardization Agent (port 8002)"
	@echo "  start-agent2         Start AI Preparation Agent (port 8003)"
	@echo "  start-agent3         Start Vector Processing Agent (port 8004)"
	@echo "  start-agent4         Start Calc Validation Agent (port 8005)"
	@echo "  start-agent5         Start QA Validation Agent (port 8006)"
	@echo "  start-agent6         Start Quality Control Agent (port 8007)"
	@echo "  start-reasoning      Start Reasoning Agent (port 8008)"
	@echo "  start-sql            Start SQL Agent (port 8009)"
	@echo "  start-agent-manager  Start Agent Manager (port 8010)"
	@echo "  start-data-manager   Start Data Manager (port 8011)"
	@echo "  start-catalog-manager Start Catalog Manager (port 8012)"
	@echo "  start-calculation    Start Calculation Agent (port 8013)"
	@echo "  start-agent-builder  Start Agent Builder (port 8014)"
	@echo "  start-embedding      Start Embedding Fine-Tuner (port 8015)"
	@echo "  start-all           Start all 16 agents (100% coverage)"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev    Deploy to development environment"
	@echo "  deploy-prod   Deploy to production environment"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,telemetry,docs]"

# Testing
test:
	pytest tests/ --cov=a2a --cov-report=term-missing

test-unit:
	pytest tests/ -m "unit" --cov=a2a

test-integration:
	pytest tests/ -m "integration"

test-quick:
	pytest tests/ -x --ff

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/a2a/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Development
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/

dev-setup: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Agent Management - All 16 Agents for 100% Coverage
start-registry:
	python3 scripts/launch/unified_launcher.py --agent=registry-server --port=8000

start-agent0:
	python3 scripts/launch/unified_launcher.py --agent=agent0 --port=8001

start-agent1:
	python3 scripts/launch/unified_launcher.py --agent=agent1 --port=8002

start-agent2:
	python3 scripts/launch/unified_launcher.py --agent=agent2 --port=8003

start-agent3:
	python3 scripts/launch/unified_launcher.py --agent=agent3 --port=8004

start-agent4:
	python3 scripts/launch/unified_launcher.py --agent=agent4 --port=8005

start-agent5:
	python3 scripts/launch/unified_launcher.py --agent=agent5 --port=8006

start-agent6:
	python3 scripts/launch/unified_launcher.py --agent=agent6 --port=8007

start-reasoning:
	python3 scripts/launch/unified_launcher.py --agent=reasoning-agent --port=8008

start-sql:
	python3 scripts/launch/unified_launcher.py --agent=sql-agent --port=8009

start-agent-manager:
	python3 scripts/launch/unified_launcher.py --agent=agent-manager --port=8010

start-data-manager:
	python3 scripts/launch/unified_launcher.py --agent=data-manager --port=8011

start-catalog-manager:
	python3 scripts/launch/unified_launcher.py --agent=catalog-manager --port=8012

start-calculation:
	python3 scripts/launch/unified_launcher.py --agent=calculation-agent --port=8013

start-agent-builder:
	python3 scripts/launch/unified_launcher.py --agent=agent-builder --port=8014

start-embedding:
	python3 scripts/launch/unified_launcher.py --agent=embedding-finetuner --port=8015

start-managers: start-data-manager start-catalog-manager start-agent-manager

start-all:
	@echo "Starting ALL 16 A2A agents for 100% coverage..."
	python3 scripts/launch/unifiedLauncher.py --agent=registry-server --port=8000 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent0 --port=8001 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent1 --port=8002 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent2 --port=8003 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent3 --port=8004 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent4 --port=8005 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent5 --port=8006 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent6 --port=8007 &
	python3 scripts/launch/unifiedLauncher.py --agent=reasoning-agent --port=8008 &
	python3 scripts/launch/unifiedLauncher.py --agent=sql-agent --port=8009 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent-manager --port=8010 &
	python3 scripts/launch/unifiedLauncher.py --agent=data-manager --port=8011 &
	python3 scripts/launch/unifiedLauncher.py --agent=catalog-manager --port=8012 &
	python3 scripts/launch/unifiedLauncher.py --agent=calculation-agent --port=8013 &
	python3 scripts/launch/unifiedLauncher.py --agent=agent-builder --port=8014 &
	python3 scripts/launch/unifiedLauncher.py --agent=embedding-finetuner --port=8015 &
	@echo "All 16 agents started in background (100% coverage). Use 'make stop-all' to stop them."

stop-all:
	@echo "Stopping all A2A agents..."
	pkill -f "unified_launcher.py" || true
	@echo "All agents stopped."

# Deployment
deploy-dev:
	python -m a2a.cli deploy --environment development

deploy-staging:
	python -m a2a.cli deploy --environment staging

deploy-prod:
	python -m a2a.cli deploy --environment production

# Monitoring
status:
	python -m a2a.cli status --all

logs:
	tail -f *.log

# Docker
docker-build:
	docker build -t a2a-network:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Database
migrate:
	python -m a2a.core.ai_decision_database_integration migrate

# Blockchain
deploy-contracts:
	python3 scripts/deployment/deploy_real_contracts.py