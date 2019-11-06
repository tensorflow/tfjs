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
  name = "unary",
  srcs = ["unary.h"],
  deps = [
    ":backend",
  ],
)

tfjs_cc_library(
  name = "binary",
  srcs = ["binary.h"],
  deps = [
    ":backend",
  ],
)

tfjs_cc_library(
  name = "transpose",
  srcs = ["transpose.h"],
  deps = [
    ":backend",
  ],
)

tfjs_cc_library(
  name = "all_kernels",
  deps = [
    ":Abs",
    ":Add",
    ":BatchMatMul",
    ":Conv2D",
    ":Div",
    ":Mul",
    ":Prelu",
    ":FusedBatchNorm",
    ":Max",
    ":Min",
    ":Sigmoid",
    ":Sub",
    ":Transpose",
  ]
)

tfjs_cc_library(
  name = "FusedBatchNorm",
  srcs = ["kernels/FusedBatchNorm.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Max",
  srcs = ["kernels/Max.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Min",
  srcs = ["kernels/Min.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Abs",
  srcs = ["kernels/Abs.cc"],
  deps = [
    ":backend",
    ":unary",
  ],
)

tfjs_cc_library(
  name = "Add",
  srcs = ["kernels/Add.cc"],
  deps = [
    ":backend",
    ":binary",
    ":util",
  ],
)

tfjs_cc_library(
  name = "BatchMatMul",
  srcs = ["kernels/BatchMatMul.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Conv2D",
  srcs = ["kernels/Conv2D.cc"],
  hdrs = ["kernels/Conv2D.h"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Div",
  srcs = ["kernels/Div.cc"],
  deps = [
    ":backend",
    ":binary",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Mul",
  srcs = ["kernels/Mul.cc"],
  deps = [
    ":backend",
    ":binary",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Prelu",
  srcs = ["kernels/Prelu.cc"],
  hdrs = ["kernels/Prelu.h"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Sigmoid",
  srcs = ["kernels/Sigmoid.cc"],
  deps = [
    ":backend",
    ":unary",
  ],
)

tfjs_cc_library(
  name = "Square",
  srcs = ["kernels/Square.cc"],
  deps = [
    ":backend",
    ":unary",
  ],
)

tfjs_cc_library(
  name = "Sub",
  srcs = ["kernels/Sub.cc"],
  deps = [
    ":backend",
    ":binary",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Transpose",
  srcs = ["kernels/Transpose.cc"],
  deps = [
    ":backend",
    ":util",
    ":transpose",
  ],
)

tfjs_cc_library(
  name = "util",
  hdrs = ["util.h"],
  srcs = ["util.cc"],
)

test_suite(
  name = "cc_tests"
)

tfjs_unit_test(
  name = "backend_tests",
  srcs = glob(["*_test.cc"]),
  deps = [
    ":backend",
    ":util",
    ":Prelu",
  ]
)

tfjs_unit_test(
  name = "Prelu_test",
  srcs = ["kernels/Prelu_test.cc"],
  deps = [
    ":Prelu",
  ]
)

tfjs_unit_test(
  name = "Conv2D_test",
  srcs = ["kernels/Conv2D_test.cc"],
  deps = [
    ":Conv2D",
  ]
)
