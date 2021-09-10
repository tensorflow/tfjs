load("@rules_python//python:defs.bzl", "py_runtime_pair")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")

package(default_visibility = ["//visibility:public"])

py_runtime(
    name = "python3_runtime",
    files = ["@python3_interpreter//:files"],
    interpreter = "@python3_interpreter//:python3_bin",
    python_version = "PY3",
    visibility = ["//visibility:public"],
)

py_runtime(
    name = "python2_runtime",
    files = ["@python2_interpreter//:files"],
    interpreter = "@python2_interpreter//:python_bin",
    python_version = "PY2",
    visibility = ["//visibility:public"],
)

py_runtime_pair(
    name = "tfjs_py_runtime_pair",
    py2_runtime = ":python2_runtime",
    py3_runtime = ":python3_runtime",
)

toolchain(
    name = "tfjs_py_toolchain",
    toolchain = ":tfjs_py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

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
        'tensorflow-hub>=0.7.0,<0.10; python_version < "3"',
        'tensorflow-hub>=0.7.0,<0.13; python_version >= "3"',
    ],
    strip_path_prefixes = [
        "tfjs-converter/python",
    ],
    version = "3.10.0",
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
        "six>=1.12.0,<2",
        'tensorflow-hub>=0.7.0,<0.10; python_version < "3"',
        'tensorflow-hub>=0.7.0,<0.13; python_version >= "3"',
    ],
    strip_path_prefixes = [
        "tfjs-converter/python",
    ],
    version = "3.10.0",
    #keywords='tensorflow javascript machine deep learning converter',
    deps = [
        ":tensorflowjs_pkg",
        "//tfjs-converter/python/tensorflowjs",
    ],
)
