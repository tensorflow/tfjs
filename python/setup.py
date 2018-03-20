# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Build pip wheel for model_converter."""

import setuptools


REQUIRED_PACKAGES = [
    'h5py >= 2.7.1',
    'keras >= 2.1.4',
    'numpy >= 1.14.1',
    'six >= 1.11.0',
]

CONSOLE_SCRIPTS = [
    'tensorflowjs_converter = tensorflowjs.converters.converter:main',
]

setuptools.setup(
    name='tensorflowjs',
    version='0.0.1',
    description='Python Libraries and Tools for TensorFlow.js',
    url='https://js.tensorflow.org/',
    py_modules=[
        'tensorflowjs',
        'tensorflowjs.write_weights',
        'tensorflowjs.converters',
        'tensorflowjs.converters.h5_conversion',
        'tensorflowjs.converters.converter',
    ],
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    license='MIT',
    keywords='tensorflow javascript machine deep learning converter',
)
