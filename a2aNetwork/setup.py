"""
Setup script for A2A Network
"""

from setuptools import setup, find_packages
import os

# Read README if available
readme_path = os.path.join(os.path.dirname(__file__), 'readme.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="a2a-network",
    version="1.0.0",
    description="A2A Network Infrastructure - Registry, Trust, and SDK Components",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="A2A Network Team",
    author_email="dev@a2a.network",
    url="https://github.com/a2a-network/a2a-network",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "httpx>=0.25.0",
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
        "redis>=5.0.0",
        "celery>=5.3.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-exporter-otlp>=1.20.0",
        "opentelemetry-instrumentation-fastapi>=0.41b0",
        "opentelemetry-instrumentation-requests>=0.41b0",
        "opentelemetry-instrumentation-sqlalchemy>=0.41b0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "blockchain": [
            "web3>=6.10.0",
            "eth-account>=0.9.0",
            "eth-keys>=0.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: System :: Distributed Computing",
    ],
    entry_points={
        "console_scripts": [
            "a2a-registry=registry.runRegistryServer:main",
            "a2a-trust=trustSystem.service:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.json", "*.sql", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)