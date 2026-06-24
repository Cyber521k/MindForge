from setuptools import setup, find_packages

setup(
    name="mindforge",
    version="0.1.0",
    description="AI model probing and correction system for DPO training data generation",
    packages=find_packages(),
    install_requires=[
        "mlx-lm",
        "openai",
        "requests",
        "pymupdf",
        "beautifulsoup4",
        "pyyaml",
        "datasets",
    ],
    entry_points={
        "console_scripts": [
            "mindforge=mindforge.cli:main",
        ],
    },
    python_requires=">=3.9",
)
