package(
    default_visibility = [
        "//tfjs-converter/python/tensorflowjs:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "converters",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":converter",
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        ":tf_saved_model_conversion_v2",
        ":wizard",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY2AND3",
    deps = ["//tfjs-converter/python/tensorflowjs:version"],
)

py_library(
    name = "keras_h5_conversion",
    srcs = ["keras_h5_conversion.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        "//tfjs-converter/python/tensorflowjs:expect_h5py_installed",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:write_weights",
    ],
)

py_test(
    name = "keras_h5_conversion_test",
    srcs = ["keras_h5_conversion_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":keras_h5_conversion",
        "//tfjs-converter/python/tensorflowjs:expect_h5py_installed",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
        "//tfjs-converter/python/tensorflowjs:version",
    ],
)

py_library(
    name = "keras_tfjs_loader",
    srcs = ["keras_tfjs_loader.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras_h5_conversion",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:read_weights",
    ],
)

py_test(
    name = "keras_tfjs_loader_test",
    srcs = ["keras_tfjs_loader_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "graph_rewrite_util",
    srcs = ["graph_rewrite_util.py"],
    srcs_version = "PY2AND3",
    deps = ["//tfjs-converter/python/tensorflowjs:version"],
)

py_library(
    name = "fuse_prelu",
    srcs = ["fuse_prelu.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_rewrite_util",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_test(
    name = "fuse_prelu_test",
    srcs = ["fuse_prelu_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":fuse_prelu",
        ":tf_saved_model_conversion_v2",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "fold_batch_norms",
    srcs = ["fold_batch_norms.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_rewrite_util",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_test(
    name = "fold_batch_norms_test",
    srcs = ["fold_batch_norms_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":fold_batch_norms",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "fuse_depthwise_conv2d",
    srcs = ["fuse_depthwise_conv2d.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":graph_rewrite_util",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_test(
    name = "fuse_depthwise_conv2d_test",
    srcs = ["fuse_depthwise_conv2d_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":fuse_depthwise_conv2d",
        ":graph_rewrite_util",
        ":tf_saved_model_conversion_v2",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_test(
    name = "tf_saved_model_conversion_v2_test",
    srcs = ["tf_saved_model_conversion_v2_test.py"],
    data = ["//tfjs-converter/python/tensorflowjs/op_list:ops"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":tf_saved_model_conversion_v2",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_hub_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "tf_saved_model_conversion_v2",
    srcs = ["tf_saved_model_conversion_v2.py"],
    data = ["//tfjs-converter/python/tensorflowjs/op_list:ops"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        ":fold_batch_norms",
        ":fuse_depthwise_conv2d",
        ":fuse_prelu",
        ":graph_rewrite_util",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_packaging_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_hub_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
        "//tfjs-converter/python/tensorflowjs:resource_loader",
        "//tfjs-converter/python/tensorflowjs:version",
        "//tfjs-converter/python/tensorflowjs:write_weights",
    ],
)

py_test(
    name = "wizard_test",
    srcs = ["wizard_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":wizard",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
    ],
)

py_binary(
    name = "wizard",
    srcs = ["wizard.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        ":converter",
        "//tfjs-converter/python/tensorflowjs:expect_PyInquirer_installed",
        "//tfjs-converter/python/tensorflowjs:expect_h5py_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_library(
    name = "converter_lib",
    srcs = ["converter.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        ":tf_saved_model_conversion_v2",
        "//tfjs-converter/python/tensorflowjs:version",
    ],
)

py_binary(
    name = "converter",
    srcs = ["converter.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":common",
        ":keras_h5_conversion",
        ":keras_tfjs_loader",
        ":tf_saved_model_conversion_v2",
        "//tfjs-converter/python/tensorflowjs:expect_h5py_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
        "//tfjs-converter/python/tensorflowjs:version",
    ],
)

py_binary(
    name = "generate_test_model",
    testonly = True,
    srcs = ["generate_test_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    imports = ["../.."],
    srcs_version = "PY2AND3",
    tags = ["ci"],
    deps = [
        ":converter",
        ":keras_tfjs_loader",
        "//tfjs-converter/python/tensorflowjs:expect_numpy_installed",
        "//tfjs-converter/python/tensorflowjs:expect_tensorflow_installed",
        "//tfjs-converter/python/tensorflowjs:version",
    ],
)

sh_test(
    name = "converter_binary_test",
    srcs = ["converter_binary_test.sh"],
    data = [
        ":converter",
        ":generate_test_model",
    ],
    tags = ["ci"],
)
