from setuptools import setup, find_packages

setup(
    name="a2a-project",
    version="0.1.0",
    packages=find_packages(exclude=['tests', 'tests.*']),
    author="A2A Team",
    description="A2A Agent and Network Infrastructure",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
