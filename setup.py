import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open("requirements/requirements.txt") as f:
    requirements = f.read()

setuptools.setup(
    name="Tonami",
    version="0.0.1",
    author="Sabrina Yu",
    author_email="sab.yu@mail.utoronto.ca",
    description="The Python package for the Tonami web application backend",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "tonami"},
    packages=[requirements],
    python_requires=">=3.9",
)