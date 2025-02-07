from setuptools import setup, find_packages
from pathlib import Path
from typing import List

def read_requirements(filename: str) -> List[str]:
    return Path(filename).read_text().splitlines()

setup(
    name="car-damage-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "httpx",
            "pytest-cov"
        ]
    },
    python_requires=">=3.8",
)