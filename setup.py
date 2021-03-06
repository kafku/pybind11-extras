#!/usr/bin/env python

# Setup script for PyPI; use CMakeFile.txt to build extension modules

from setuptools import setup
from distutils.command.install_headers import install_headers
from distutils.command.build_py import build_py
from pybind11_extras import __version__
import os

package_data = [
    'include/pybind11-extras/afarray.h'
]

# Prevent installation of pybind11 headers by setting
# PYBIND11_USE_CMAKE.
if os.environ.get('PYBIND11_USE_CMAKE'):
    headers = []
else:
    headers = package_data


class InstallHeaders(install_headers):
    """Use custom header installer because the default one flattens subdirectories"""
    def run(self):
        if not self.distribution.headers:
            return

        for header in self.distribution.headers:
            subdir = os.path.dirname(os.path.relpath(header, 'include/pybind11-extras'))
            install_dir = os.path.join(self.install_dir, subdir)
            self.mkpath(install_dir)

            (out, _) = self.copy_file(header, install_dir)
            self.outfiles.append(out)


# Install the headers inside the package as well
class BuildPy(build_py):
    def build_package_data(self):
        build_py.build_package_data(self)
        for header in package_data:
            target = os.path.join(self.build_lib, 'pybind11-extras', header)
            self.mkpath(os.path.dirname(target))
            self.copy_file(header, target, preserve_mode=False)


setup(
    name='pybind11-extras',
    version=__version__,
    description='Extra type casters for pybind11',
    author='Kazuki Fukui',
    author_email='kazunoko93@gmail.com',
    url='https://github.com/kafku/pybind11-extras',
    download_url='https://github.com/kafku/pybind11-extras/tarball/v' + __version__,
    packages=['pybind11_extras'],
    install_requires=[
        'pybind11>=2.2.4'
    ],
    license='BSD',
    headers=headers,
    zip_safe=False,
    cmdclass=dict(install_headers=InstallHeaders, build_py=BuildPy),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License'
    ],
    keywords='C++11, Python bindings, pybind11, arrayfire',
    long_description="""

    """)
