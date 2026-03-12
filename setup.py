"""
Setup module adapted from setuptools code. See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
	name='pyNBS',
	version='0.2.0',
	description='Python package to perform network based stratification of binary somatic mutations as described in Hofree et al 2013.',
	url='https://github.com/huangger/pyNBS',
	author='Justin Huang',
	author_email='jkh013@ucsd.edu',
	license='MIT',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Topic :: Software Development :: Build Tools',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3 :: Only'
	],
	python_requires='>=3.6',
	packages=find_packages(exclude=['os', 'random', 'time']),
	install_requires=[
		'lifelines>=0.27.0',
		'networkx>=2.0',
		'numpy>=1.18.0',
		'matplotlib>=3.0.0',
		'pandas>=1.0.0',
		'scipy>=1.4.0',
		'scikit-learn>=0.22.0',
		'seaborn>=0.10.0',
		'requests>=2.20.0']
)
