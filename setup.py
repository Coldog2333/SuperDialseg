#!/usr/bin/env python3

from setuptools import setup, find_packages


setup(
    name='super_dialseg',
    version='0.0.2',
    description='Supervised Dialogue Segmentation',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license='MIT',
    author='Coldog',
    author_email='jiangjf@is.s.u-tokyo.ac.jp',
    packages=find_packages('src'),
    package_dir={'': 'src'},

    include_package_data=True,

    install_requires=[
        'torch',
        'transformers',
        'prettytable',
        'pytorch-lightning==1.9.1',
        'scikit-learn',
        'tqdm',
        'nltk==3.8.1',
        'ipdb',
        'sentencepiece',
        'wandb'
    ],

    extras_require={
    },

    python_requires='>=3.9',
    zip_safe=False,
)
