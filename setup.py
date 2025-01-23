from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="onebatchpam",
    version="0.1.0",
    description="OneBatchPAM",
    author="Antoine de Mathelin",
    author_email="antoine.demat@gmail.com",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repository",
    packages=find_packages(),
    ext_modules = cythonize("onebatchpam/onebatchpam.pyx", annotate=True),
    install_requires=["numpy>=1.19.0", "Cython>=0.29"],
)