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

py_binary(
    name = "h5_conversion",
    srcs = ["h5_conversion.py"],
    deps = [
        ":expect_h5py_installed",
        ":expect_keras_installed",
        ":expect_numpy_installed",
    ],
)

py_test(
    name = "h5_conversion_test",
    srcs = ["h5_conversion_test.py"],
    deps = [
        ":expect_h5py_installed",
        ":expect_keras_installed",
        ":expect_numpy_installed",
        ":h5_conversion",
    ],
)

py_binary(
    name = "keras_model_converter",
    srcs = ["keras_model_converter.py"],
    deps = [
        ":expect_h5py_installed",
        ":h5_conversion",
    ],
)

py_test(
    name = "keras_model_converter_test",
    srcs = ["keras_model_converter_test.py"],
    deps = [
        ":expect_keras_installed",
        ":pykeras",
    ],
)

py_binary(
    name = "benchmarks",
    srcs = ["benchmarks.py"],
    deps = [
        ":expect_keras_installed",
        ":expect_numpy_installed",
        ":h5_conversion",
    ],
)

py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":benchmarks",
        ":expect_h5py_installed",
        ":expect_keras_installed",
        ":expect_numpy_installed",
        ":h5_conversion",
    ],
)

py_library(
    name = "iris_data",
    srcs = ["iris_data.py"],
    deps = [
        ":expect_numpy_installed",
    ],
)

py_test(
    name = "iris_data_test",
    srcs = ["iris_data_test.py"],
    deps = [
        ":expect_numpy_installed",
        ":iris_data",
    ],
)

py_binary(
    name = "iris",
    srcs = ["iris.py"],
    deps = [
        ":expect_keras_installed",
        ":expect_numpy_installed",
        ":h5_conversion",
        ":iris_data",
    ],
)

py_test(
    name = "iris_test",
    srcs = ["iris_test.py"],
    deps = [
        ":iris",
    ],
)

py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        ":expect_keras_installed",
        ":h5_conversion",
    ],
)

py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    deps = [
        ":expect_numpy_installed",
        ":mnist",
    ],
)

py_binary(
    name = "mnist_transfer_cnn",
    srcs = ["mnist_transfer_cnn.py"],
    deps = [
        ":expect_keras_installed",
        ":h5_conversion",
    ],
)

py_test(
    name = "mnist_transfer_cnn_test",
    srcs = ["mnist_transfer_cnn_test.py"],
    deps = [
        ":expect_numpy_installed",
        ":mnist_transfer_cnn",
    ],
)

py_binary(
    name = "mobilenet",
    srcs = ["mobilenet.py"],
    data = [
        "imagenet_class_names.json",
    ],
    deps = [
        ":expect_h5py_installed",
        ":expect_keras_installed",
        ":h5_conversion",
    ],
)

py_binary(
    name = "translation",
    srcs = ["translation.py"],
    deps = [
        ":expect_h5py_installed",
        ":expect_keras_installed",
        ":h5_conversion",
    ],
)

# TODO(cais): Add unit test target for translation.py: translation_test.
