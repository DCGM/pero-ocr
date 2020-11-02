#!/usr/bin/env python3

from setuptools import setup


def get_long_desc():
    with open('README.md') as f:
        return f.read()


setup(
    name='pero-ocr',
    version='0.4',
    python_requires='>=3.6',
    packages=[
        'pero_ocr',
        'pero_ocr/decoding',
        'pero_ocr/document_ocr',
        'pero_ocr/layout_engines',
        'pero_ocr/line_engine',
        'pero_ocr/ocr_engine',
        'pero_ocr/region_engine',
    ],
    install_requires=[
        'numpy',
        'opencv-python',
        'lxml',
        'scipy',
        'numba',
        'torch>=1.4',
        'brnolm>=0.1.1',
        'scikit-learn',
        'scikit-image',
        'tensorflow-gpu==1.15',
        'shapely',
        'pyamg',
        'imgaug',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    url='https://github.com/DCGM/pero-ocr',
    description='Toolkit for advanced OCR of poor quality documents',
    long_description=get_long_desc(),
    long_description_content_type='text/markdown',
    author='Karel Benes',
    author_email='ibenes@fit.vutbr.cz',
)
