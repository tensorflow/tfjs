py_library(
    name = "keras_h5_conversion",
    srcs = ["keras_h5_conversion.py"],
    deps = [
        "//tensorflowjs:expect_h5py_installed",
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
        "//tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "keras_tfjs_loader",
    srcs = ["keras_tfjs_loader.py"],
    deps = [
        ":keras_h5_conversion",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_numpy_installed",
        "//tensorflowjs:read_weights",
    ],
)

py_test(
    name = "keras_tfjs_loader_test",
    srcs = ["keras_tfjs_loader_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_numpy_installed",
        "//tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "tf_saved_model_conversion",
    srcs = ["tf_saved_model_conversion.py"],
    data = ["//tensorflowjs:op_list_jsons"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflowjs:expect_numpy_installed",
        "//tensorflowjs:expect_tensorflow_installed",
        "//tensorflowjs:write_weights",
    ],
)

py_test(
    name = "tf_saved_model_conversion_test",
    srcs = ["tf_saved_model_conversion_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tf_saved_model_conversion",
        "//tensorflowjs:expect_tensorflow_installed",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        ":tf_saved_model_conversion",
        "//tensorflowjs:expect_h5py_installed",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_tensorflow_installed",
        "//tensorflowjs:version",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":converter",
        "//tensorflowjs:expect_keras_installed",
        "//tensorflowjs:expect_tensorflow_installed",
    ],
)
