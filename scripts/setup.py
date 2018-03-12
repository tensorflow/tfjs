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
    'keras_model_converter = keras_model_converter:main',
]

setuptools.setup(
    name='keras_model_converter',
    version='0.1.0',
    py_modules=['h5_conversion', 'keras_model_converter', 'write_weights'],
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
)
