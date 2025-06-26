from setuptools import setup,find_packages

with open("requirements.txt") as f:
   requirements = f.read().splitlines()

setup(
   name="MLOPS-Project4",
   version="0.1",
   author="Siddhesh",
   packages= find_packages(),
   install_requires= requirements,
)