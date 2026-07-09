from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="odfe-obm-oacl",
    version="1.0.0",
    description="Optimization-Driven Feature Learning for Long-Tailed Object Detection in Optical Remote Sensing Imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nishi Madaan, Rahul Malik, Utsav Upadhyay, Alok Kumar, Satyabrata Roy, Umashankar Rawat",
    author_email="rahul.malik@galgotiasuniversity.edu.in",
    url="https://github.com/nishimaliknitj/ODFE-OBM-OACL-RSDetection",
    license="MIT",
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python>=4.7.0",
        "numpy>=1.23.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.6.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort", "pytest", "pytest-cov", "pre-commit"],
    },
    entry_points={
        "console_scripts": [
            "odfe-train=train:main",
            "odfe-test=test:main",
            "odfe-predict=predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
