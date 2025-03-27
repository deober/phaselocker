from setuptools import setup

__version__ = "0.1"


setup(
    name="phaselocker",
    author="Derick Ober",
    version=__version__,
    packages=["phaselocker"],
    install_requires=["numpy", "scipy", "bokeh"],
)
