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
        "flax>=0.5.3",
        "importlib_resources>=5.9.0",
        "jax>=0.3.16",
        "protobuf<3.20,>=3.9.2",
        "tensorflow>=2.10.0,<3",
        "tensorflow-decision-forests>=1.0.1",
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
