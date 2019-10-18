load(":build_defs.bzl", "tfjs_cc_library", "tfjs_unit_test")

# Emcripten produces a much larger wasm bundle unless the cc_binary has srcs
# explicitly pointing to files with exported methods (EMSCRIPTEN_KEEPALIVE).
KERNELS_WITH_KEEPALIVE = glob(
  ["kernels/*.cc"],
  exclude = ["**/*_test.cc"],
)

cc_binary(
    name = "tfjs-backend-wasm.js",
    srcs = ['backend.cc'] + KERNELS_WITH_KEEPALIVE,
    linkopts = [
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        "-s FILESYSTEM=0",
        "-s EXIT_RUNTIME=0",
        "-s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'",
        "-s EXTRA_EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
        "-s ENVIRONMENT=web",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=WasmBackendModule",
        "-s MALLOC=emmalloc",
    ],
    deps = [
        ":backend",
        ":all_kernels",
    ],
)

tfjs_cc_library(
  name = "backend",
  srcs = ["backend.cc"],
  hdrs = ["backend.h"],
  deps = [
    ":util",
    "@xnnpack//:xnnpack_operators_nhwc_f32",
  ],
)

tfjs_cc_library(
  name = "all_kernels",
  deps = [
    ":add",
    ":batch_matmul",
    ":prelu",
  ]
)

tfjs_cc_library(
  name = "add",
  srcs = ["kernels/add.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "batch_matmul",
  srcs = ["kernels/batch_matmul.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "prelu",
  srcs = ["kernels/prelu.cc"],
  hdrs = ["kernels/prelu.h"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "util",
  srcs = ["util.h"],
)

test_suite(
  name = "cc_tests"
)

tfjs_unit_test(
  name = "backend_test",
  srcs = ["backend_test.cc"],
  deps = [
    ":backend",
    ":prelu",
  ]
)

tfjs_unit_test(
  name = "prelu_test",
  srcs = ["kernels/prelu_test.cc"],
  deps = [
    ":prelu",
  ]
)
