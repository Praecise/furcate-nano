# ============================================================================
# setup.py
"""Setup configuration for Furcate Nano."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="furcate-nano",
    version="1.0.0",
    author="Furcate Team",
    author_email="opensource@furcate.org",
    description="Open Source Environmental Edge Computing Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/furcate-team/furcate-nano",
    project_urls={
        "Documentation": "https://docs.furcate-nano.org",
        "Source": "https://github.com/furcate-team/furcate-nano",
        "Tracker": "https://github.com/furcate-team/furcate-nano/issues",
        "Community": "https://discord.gg/furcate-nano",
        "Furcate Ecosystem": "https://furcate.org"
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers", 
        "Topic :: Scientific/Engineering :: Environmental Science",
        "Topic :: System :: Hardware",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
        "Environment :: No Input/Output (Daemon)",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950"
        ],
        "ml": [
            "tensorflow-lite>=2.10.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0"
        ],
        "full": [
            "opencv-python>=4.5.0",
            "pillow>=8.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "furcate-nano=furcate_nano.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "furcate_nano": [
            "configs/*.yaml",
            "models/*.tflite",
            "scripts/*.sh"
        ]
    },
    zip_safe=False,
)