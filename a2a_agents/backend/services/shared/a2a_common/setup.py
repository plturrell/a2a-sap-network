"""
A2A Common Library - Shared components for all A2A agents
"""
from setuptools import setup, find_packages

setup(
    name="a2a-common",
    version="0.2.9",
    description="A2A (Agent-to-Agent) Common Library - SDK, Core, Skills, Security",
    author="A2A Team",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "pydantic>=2.5.0",
        "httpx>=0.25.0",
        "prometheus-client>=0.19.0",
        "PyYAML>=6.0",
        "cryptography>=41.0.0",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)