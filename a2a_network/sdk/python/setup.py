from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="a2a-network-sdk",
    version="1.0.0",
    author="A2A Network Team",
    author_email="dev@a2a.network",
    description="Official Python SDK for A2A Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a2a-network/sdk-python",
    project_urls={
        "Bug Tracker": "https://github.com/a2a-network/sdk-python/issues",
        "Documentation": "https://docs.a2a.network",
        "Homepage": "https://a2a.network",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: System :: Networking",
        "Topic :: Security :: Cryptography",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "responses>=0.23.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "a2a-cli=a2a.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "a2a": ["constants/*.json", "templates/*.json"],
    },
    keywords=[
        "a2a", "agent", "network", "blockchain", "web3", 
        "ethereum", "messaging", "ai", "sdk", "crypto"
    ],
    zip_safe=False,
)