load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

CONSOLE_SCRIPTS = {
    "tensorflowjs_converter": "tensorflowjs.converters.converter:pip_main",
    "tensorflowjs_wizard": "tensorflowjs.converters.wizard:pip_main",
}

py_package(
    name = "tensorflowjs_pkg",
    # Only include these Python packages.
    packages = ["tfjs-converter/python/tensorflowjs"],
    deps = ["//tfjs-converter/python/tensorflowjs"],
)

py_wheel(
    name = "python3_wheel",
    author = "Google LLC",
    author_email = "opensource@google.com",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: JavaScript",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    console_scripts = CONSOLE_SCRIPTS,
    description_file = ":README.md",
    distribution = "tensorflowjs",
    extra_requires = {
        "PyInquirer": ["PyInquirer==1.0.3"],
        "all": ["PyInquirer==1.0.3"],
        "wizard": ["PyInquirer==1.0.3"],
    },
    homepage = "https://js.tensorflow.org/",
    license = "Apache 2.0",
    python_tag = "py3",
    requires = [
        "tensorflow>=2.1.0,<3",
        "six>=1.12.0,<2",
        "tensorflow-hub>=0.7.0,<0.13",
        "packaging~=20.9",
    ],
    strip_path_prefixes = [
        "tfjs-converter/python",
    ],
    version = "0.0.0",
    #keywords='tensorflow javascript machine deep learning converter',
    deps = [
        ":tensorflowjs_pkg",
        "//tfjs-converter/python/tensorflowjs",
    ],
)

py_wheel(
    name = "python2_wheel",
    author = "Google LLC",
    author_email = "opensource@google.com",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: JavaScript",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    console_scripts = CONSOLE_SCRIPTS,
    description_file = ":README.md",
    distribution = "tensorflowjs",
    extra_requires = {
        "PyInquirer": ["PyInquirer==1.0.3"],
        "all": ["PyInquirer==1.0.3"],
        "wizard": ["PyInquirer==1.0.3"],
    },
    homepage = "https://js.tensorflow.org/",
    license = "Apache 2.0",
    python_tag = "py2",
    requires = [
        "tensorflow>=2.1.0,<3",
        "protobuf==3.17.3",
        "six>=1.12.0,<2",
        "tensorflow-hub>=0.7.0,<0.10",
        "regex<2022.1.18",
        "grpcio==1.39.0",
        "packaging~=20.9",
    ],
    strip_path_prefixes = [
        "tfjs-converter/python",
    ],
    version = "0.0.0",
    #keywords='tensorflow javascript machine deep learning converter',
    deps = [
        ":tensorflowjs_pkg",
        "//tfjs-converter/python/tensorflowjs",
    ],
)
