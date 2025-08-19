from setuptools import setup, find_packages

setup(
    name="a2a-project",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    author="A2A Team",
    description="A2A Agent and Network Infrastructure",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
