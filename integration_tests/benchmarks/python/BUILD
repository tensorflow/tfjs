
# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================

py_library(
    name = "expect_keras_installed",
    # This is a dummy rule used as a keras dependency in open-source.
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
    name = "benchmarks",
    srcs = ["benchmarks.py"],
    deps = [
        ":expect_keras_installed",
        ":expect_numpy_installed",
    ],
)

py_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":benchmarks",
        ":expect_keras_installed",
        ":expect_numpy_installed",
    ],
)
