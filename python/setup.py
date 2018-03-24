# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Build pip wheel for model_converter."""

import setuptools
from tensorflowjs import version

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
    version=version.version,
    description='Python Libraries and Tools for TensorFlow.js',
    url='https://js.tensorflow.org/',
    author='Google LLC',
    author_email='opensource@google.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: JavaScript',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    py_modules=[
        'tensorflowjs',
        'tensorflowjs.version',
        'tensorflowjs.write_weights',
        'tensorflowjs.converters',
        'tensorflowjs.converters.converter',
        'tensorflowjs.converters.keras_h5_conversion',
        'tensorflowjs.converters.tf_saved_model_conversion',
    ],
    include_package_data=True,
    packages=['tensorflowjs/op_list'],
    package_data={
        'tensorflowjs/op_list': ['*.json']
    },
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    license='MIT',
    keywords='tensorflow javascript machine deep learning converter',
)
