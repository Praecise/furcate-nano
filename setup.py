from setuptools import setup, find_packages

setup(
    name="furcate-nano",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "torch>=2.0.0",
        "tensorflow>=2.15.0",
    ],
    entry_points={
        'console_scripts': [
            'furcate-nano=furcate_nano.cli:cli',
        ],
    },
)
