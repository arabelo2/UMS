from setuptools import setup, find_packages
import os

# Read the long description from README.md if available
this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")
try:
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="UMS",
    version="1.0.0",
    author="Alexandre RABELO",  # Replace with the actual author's name
    author_email="arabelo@usp.br",  # Replace with the actual author's email
    description="Ultrasonic Measurements Simulator",  # Provide a short description of your package
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arabelo2/UMS",
    packages=find_packages(where="src"),  # Automatically find packages in the 'src' directory
    package_dir={"": "src"},  # Tell setuptools that packages are under 'src'
    include_package_data=True,  # Include additional files as specified in MANIFEST.in (if any)
    install_requires=[
         "numpy",
         "scipy",
         # You can add more dependencies as your project grows
    ],
    entry_points={
         "console_scripts": [
             # Adjust the module path if your CLI entry is defined elsewhere.
             # This assumes that you have a function named 'main' in 'ums/__main__.py'
             "ums=ums.__main__:main",
         ],
    },
    classifiers=[
         "Programming Language :: Python :: 3",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: MIT License",  # Change if a different license applies
         "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
