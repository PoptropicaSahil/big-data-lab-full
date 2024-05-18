from setuptools import setup, find_packages

setup(
    name="Risk-evaluation-Story",
    version="0.1.0",
    author="Sahil Girhepuje, Revati Sawant, Mrityunjay Shukla",
    author_email="ed19b048@smail.iitm.ac.in",
    description="Evalute your risk based on certain given features ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/r-emerald/Big-Data-Lab-Project",
    packages=find_packages(),
    install_requires=open("app/requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)