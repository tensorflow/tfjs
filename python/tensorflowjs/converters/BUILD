py_library(
    name = "keras_h5_conversion",
    srcs = ["keras_h5_conversion.py"],
    deps = [
        "//tensorflowjs:expect_h5py_installed",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_numpy_installed",
        "//tensorflowjs:write_weights",
    ],
)

py_test(
    name = "keras_h5_conversion_test",
    srcs = ["keras_h5_conversion_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_h5_conversion",
        "//tensorflowjs:expect_h5py_installed",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_numpy_installed",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_h5_conversion",
        "//tensorflowjs:expect_h5py_installed",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converter",
        "//tensorflowjs:expect_keras_installed",
    ],
)
