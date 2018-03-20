package(
    default_visibility = [
        "//tensorflowjs:__subpackages__",
    ],
)

py_library(
    name = "expect_h5py_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect h5py to already be installed on the system, e.g. via
    # `pip install h5py`.
)

py_library(
    name = "expect_keras_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect keras to already be installed on the system, e.g. via
    # `pip install keras`.
)

py_library(
    name = "expect_numpy_installed",
    # This is a dummy rule used as a numpy dependency in open-source.
    # We expect numpy to already be installed on the system, e.g. via
    # `pip install numpy`.
)

py_library(
    name = "write_weights",
    srcs = ["write_weights.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":expect_numpy_installed",
    ],
)
