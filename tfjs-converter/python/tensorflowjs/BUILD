load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@tensorflowjs_dev_deps//:requirements.bzl", "requirement")

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
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":resource_loader",
        ":version",
        ":write_weights",
        "//tfjs-converter/python/tensorflowjs/converters",
    ],
)

py_library(
    name = "expect_flax_installed",
    # This is a dummy rule used as a Flax dependency in open-source.
    # We expect JAX to already be installed on the system, e.g. via
    # `pip install flax`.
    deps = [requirement("flax")],
)

py_library(
    name = "expect_h5py_installed",
    # This is a dummy rule used as a h5py dependency in open-source.
    # We expect h5py to already be installed on the system, e.g. via
    # `pip install h5py`.
    deps = [requirement("h5py")],
)

py_library(
    name = "expect_jax_installed",
    # This is a dummy rule used as a JAX dependency in open-source.
    # We expect JAX to already be installed on the system, e.g. via
    # `pip install jax`.
    deps = [
        requirement("jax"),
        requirement("importlib_resources"),
    ],
)

py_library(
    name = "expect_jax2tf_installed",
    # This is a dummy rule used as a jax2tf dependency in open-source.
    # We expect jax2tf to already be installed on the system, e.g. via
    # `pip install jax`.
    deps = [requirement("jax")],
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
    name = "expect_tensorflow_decision_forests_installed",
    # This is a dummy rule used as a tensorflow dependency in open-source.
    # We expect tensorflow-decision-forests to already be installed on
    # the system, e.g. via
    # `pip install tensorflow-decision-forests`.
    deps = [requirement("tensorflow-decision-forests")],
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
    srcs_version = "PY3",
    deps = [
        ":expect_numpy_installed",
    ],
)

py_library(
    name = "write_weights",
    srcs = ["write_weights.py"],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":expect_numpy_installed",
        ":quantization",
    ],
)

py_library(
    name = "resource_loader",
    srcs = ["resource_loader.py"],
    srcs_version = "PY3",
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    imports = [".."],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    tags = ["ci"],
    deps = [
        ":resource_loader",
    ],
)
