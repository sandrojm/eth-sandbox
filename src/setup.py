from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["helpers", "ehr_pipeline"],
)
