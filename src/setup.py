from setuptools import setup
from setuptools.dist import Distribution
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(os.path.join(here, os.path.pardir), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False
setup(
    name = 'hexapy',
    version = '0.0.3',
    description = 'Provides tools for easy generation of hexahedral meshes of primitive shapes: boxes, elliptical cylinders, and ellipsoids, for use in finite element models.',
    packages = ['hexapy'],
    long_description = long_description,
    url = "https://github.com/siboles/hexapy",
    author = 'Scott Sibole',
    author_email = 'scott.sibole@gmail.com',
    license = 'GPL',
    install_requires = ['numpy', 'scipy'],
    py_modules = ['hexapy.__init__', 'hexapy.hexapy'],
    distclass=BinaryDistribution,
)
