from setuptools import setup, find_packages

setup(
    name="UMS",
    version="0.1.0",
    packages=find_packages(where="src"),  # Automatically find packages in the `src` directory
    package_dir={"": "src"},  # Tell setuptools that packages are under `src`
    install_requires=[  # List your project dependencies here
        "numpy",
        "scipy",
        # Add other dependencies as needed
    ],
    python_requires=">=3.8",  # Specify the Python version requirements
)