from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ctrl-world",
    version="0.1.0",
    author="Your Name",  # Update with your name
    author_email="your.email@example.com",  # Update with your email
    description="Ctrl-World: A world model for robotic control and manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/riccardofeingold/Ctrl-World",
    packages=find_packages(exclude=["scripts", "output", "swanlog", "wandb", "synthetic_traj", "__pycache__"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  # Update based on your LICENSE.txt
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ctrl-world-train=scripts.train_wm:main",
            "ctrl-world-inference=scripts.inference_wm:main",
        ],
    },
    include_package_data=True,
)
