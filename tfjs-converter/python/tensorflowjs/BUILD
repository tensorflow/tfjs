load("@//:python_toolchain.bzl", requirement = "dev_requirement")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = [
        ":__subpackages__",
    ],
)

licenses(["notice"])

exports_files([
    "LICENSE",
])

py_library(
    name = "tensorflowjs",
    srcs = ["__init__.py"],
    data = [
        "//tfjs-converter/python/tensorflowjs/op_list:ops",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":resource_loader",
        ":version",
        ":write_weights",
        "//tfjs-converter/python/tensorflowjs/converters",
    ],
)

py_library(
    name = "expect_h5py_installed",
    # This is a dummy rule used as a h5py dependency in open-source.
    # We expect h5py to already be installed on the system, e.g. via
    # `pip install h5py`.
    deps = [requirement("h5py")],
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`.
    deps = [requirement("numpy")],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a tensorflow dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow-gpu`.
    deps = [requirement("tensorflow")],
)

py_library(
    name = "expect_tensorflow_hub_installed",
    # This is a dummy rule used as a tensorflow_hub dependency in open-source.
    # We expect tensorflow to already be installed on the system, e.g. via
    # `pip install tensorflow` or `pip install tensorflow-gpu`.
    deps = [requirement("tensorflow-hub")],
)

py_library(
    name = "expect_PyInquirer_installed",
    # This is a dummy rule used as a PyInquirer dependency in open-source.
    # We expect PyInquirer to already be installed on the system, e.g. via
    # `pip install PyInquirer`.
    deps = [requirement("PyInquirer")],
)

py_library(
    name = "expect_packaging_installed",
    deps = [requirement("packaging")],
)

py_library(
    name = "quantization",
    srcs = ["quantization.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
    ],
)

py_library(
    name = "write_weights",
    srcs = ["write_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
        ":expect_tensorflow_installed",
        ":quantization",
        ":read_weights",
    ],
)

py_library(
    name = "read_weights",
    srcs = ["read_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
        ":quantization",
    ],
)

py_library(
    name = "resource_loader",
    srcs = ["resource_loader.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    imports = [".."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":expect_numpy_installed",
        ":quantization",
    ],
)

py_test(
    name = "write_weights_test",
    srcs = ["write_weights_test.py"],
    imports = [".."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":expect_numpy_installed",
        ":expect_tensorflow_installed",
        ":write_weights",
    ],
)

py_test(
    name = "read_weights_test",
    srcs = ["read_weights_test.py"],
    imports = [".."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":expect_numpy_installed",
        ":expect_tensorflow_installed",
        ":read_weights",
        ":write_weights",
    ],
)

py_test(
    name = "resource_loader_test",
    srcs = ["resource_loader_test.py"],
    data = ["//tfjs-converter/python/tensorflowjs/op_list:ops"],
    imports = [".."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":resource_loader",
    ],
)
