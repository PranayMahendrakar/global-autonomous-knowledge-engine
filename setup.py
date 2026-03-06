"""
Setup configuration for the Global Autonomous Knowledge Engine (GAKE).
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="global-autonomous-knowledge-engine",
    version="1.0.0",
    author="Pranay M Mahendrakar",
    author_email="pranay@sonytech.in",
    description="An AI system that continuously learns about the world automatically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PranayMahendrakar/global-autonomous-knowledge-engine",
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gake=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gake": ["config/*.yaml"],
    },
)
