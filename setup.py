from setuptools import setup, find_packages
from pathlib import Path

def read_requirements(filename: str) -> List[str]:
    return Path(filename).read_text().splitlines()

setup(
    name="car-damage-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)