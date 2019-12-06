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
  hdrs = ["binary.h"],
  srcs = ["binary.cc"],
  deps = [
    ":backend",
  ],
)

tfjs_cc_library(
  name = "transpose_impl",
  hdrs = ["transpose_impl.h"],
  srcs = ["transpose_impl.cc"],
  deps = [":util"],
)

tfjs_cc_library(
  name = "prelu_impl",
  hdrs = ["prelu_impl.h"],
  srcs = ["prelu_impl.cc"],
  deps = [
    ":backend",
    ":util"
  ],
)

tfjs_cc_library(
  name = "interpolate_bilinear_impl",
  hdrs = ["interpolate_bilinear_impl.h"],
  srcs = ["interpolate_bilinear_impl.cc"],
  deps = [
    ":backend",
    ":util"
  ],
)

tfjs_cc_library(
  name = "conv2d_impl",
  hdrs = ["conv2d_impl.h"],
  srcs = ["conv2d_impl.cc"],
  deps = [
    ":backend",
    ":transpose_impl",
    ":prelu_impl",
    ":util",
  ],
)

tfjs_cc_library(
  name = "clamp_impl",
  hdrs = ["clamp_impl.h"],
  srcs = ["clamp_impl.cc"],
  deps = [
    ":backend",
    ":util"
  ],
)

tfjs_cc_library(
  name = "all_kernels",
  deps = [
    ":Abs",
    ":Add",
    ":AvgPool",
    ":AddN",
    ":ArgMax",
    ":BatchMatMul",
    ":MaxPool",
    ":ClipByValue",
    ":CropAndResize",
    ":Conv2D",
    ":DepthwiseConv2dNative",
    ":FloorDiv",
    ":FusedConv2D",
    ":FusedDepthwiseConv2D",
    ":Div",
    ":Mul",
    ":NonMaxSuppressionV3",
    ":PadV2",
    ":Prelu",
    ":Relu",
    ":Relu6",
    ":ResizeBilinear",
    ":FusedBatchNorm",
    ":Max",
    ":Min",
    ":Sigmoid",
    ":Sub",
    ":Transpose",
  ]
)

tfjs_cc_library(
  name = "MaxPool",
  srcs = ["kernels/MaxPool.cc"],
  hdrs = ["kernels/MaxPool.h"],
  deps = [
    ":backend",
    ":util"
  ]
)

tfjs_cc_library(
  name = "AvgPool",
  srcs = ["kernels/AvgPool.cc"],
  hdrs = ["kernels/AvgPool.h"],
  deps = [
    ":backend",
    ":util"
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
  name = "AddN",
  srcs = ["kernels/AddN.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "ArgMax",
  srcs = ["kernels/ArgMax.cc"],
  deps = [
    ":backend",
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
  name = "ClipByValue",
  hdrs = ["kernels/ClipByValue.h"],
  srcs = ["kernels/ClipByValue.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "CropAndResize",
  srcs = ["kernels/CropAndResize.cc"],
  deps = [
    ":backend",
    ":interpolate_bilinear_impl",
    ":util",
  ],
)

tfjs_cc_library(
  name = "ResizeBilinear",
  srcs = ["kernels/ResizeBilinear.cc"],
  deps = [
    ":backend",
    ":interpolate_bilinear_impl",
    ":util",
  ],
)

tfjs_cc_library(
  name = "Conv2D",
  srcs = ["kernels/Conv2D.cc"],
  hdrs = ["kernels/Conv2D.h"],
  deps = [
    ":conv2d_impl",
  ],
)

tfjs_cc_library(
  name = "FloorDiv",
  srcs = ["kernels/FloorDiv.cc"],
  deps = [
    ":backend",
    ":binary",
    ":util",
  ],
)

tfjs_cc_library(
  name = "FusedConv2D",
  srcs = ["kernels/FusedConv2D.cc"],
  hdrs = ["kernels/FusedConv2D.h"],
  deps = [
    ":conv2d_impl",
  ],
)

tfjs_cc_library(
  name = "FusedDepthwiseConv2D",
  srcs = ["kernels/FusedDepthwiseConv2D.cc"],
  hdrs = ["kernels/FusedDepthwiseConv2D.h"],
  deps = [
    ":conv2d_impl",
  ],
)

tfjs_cc_library(
  name = "DepthwiseConv2dNative",
  srcs = ["kernels/DepthwiseConv2dNative.cc"],
  hdrs = ["kernels/DepthwiseConv2dNative.h"],
  deps = [
    ":conv2d_impl",
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
  name = "NonMaxSuppressionV3",
  srcs = ["kernels/NonMaxSuppressionV3.cc"],
  deps = [
    ":backend",
    ":util",
  ],
)

tfjs_cc_library(
  name = "PadV2",
  srcs = ["kernels/PadV2.cc"],
  deps = [
    ":backend",
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
    ":prelu_impl",
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
  name = "Relu",
  srcs = ["kernels/Relu.cc"],
  deps = [
    ":backend",
    ":clamp_impl",
    ":unary",
  ],
)

tfjs_cc_library(
  name = "Relu6",
  srcs = ["kernels/Relu6.cc"],
  deps = [
    ":backend",
    ":clamp_impl",
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
    ":transpose_impl",
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

tfjs_unit_test(
  name = "DepthwiseConv2dNative_test",
  srcs = ["kernels/DepthwiseConv2dNative_test.cc"],
  deps = [
    ":DepthwiseConv2dNative",
  ]
)

tfjs_unit_test(
  name = "FusedConv2D_test",
  srcs = ["kernels/FusedConv2D_test.cc"],
  deps = [
    ":FusedConv2D",
  ]
)

tfjs_unit_test(
  name = "FusedDepthwiseConv2D_test",
  srcs = ["kernels/FusedDepthwiseConv2D_test.cc"],
  deps = [
    ":FusedDepthwiseConv2D",
  ]
)

tfjs_unit_test(
  name = "AvgPool_test",
  srcs = ["kernels/AvgPool_test.cc"],
  deps = [
    ":AvgPool"
  ]
)

tfjs_unit_test(
  name = "MaxPool_test",
  srcs = ["kernels/MaxPool_test.cc"],
  deps = [
    ":MaxPool"
  ]
)

tfjs_unit_test(
  name = "ClipByValue_test",
  srcs = ["kernels/ClipByValue_test.cc"],
  deps = [
    ":ClipByValue",
  ]
)
