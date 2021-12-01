load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")
load(":build_defs.bzl", "tfjs_cc_library", "tfjs_unit_test")

# Emcripten produces a much larger wasm bundle unless the cc_binary has srcs
# explicitly pointing to files with exported methods (EMSCRIPTEN_KEEPALIVE).
KERNELS_WITH_KEEPALIVE = glob(
    ["kernels/*.cc"],
    exclude = ["**/*_test.cc"],
)

BASE_LINKOPTS = [
    "-s ALLOW_MEMORY_GROWTH=1",
    "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
    "-s DISABLE_EXCEPTION_CATCHING=1",
    "-s FILESYSTEM=0",
    "-s EXIT_RUNTIME=0",
    "-s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'",
    "-s EXTRA_EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
    "-s MODULARIZE=1",
    "-s MALLOC=emmalloc",
    # TODO(mattsoulanille): Mark these files as dependencies so Bazel rebuilds
    # when they change. github.com/emscripten-core/emsdk/issues/933
    "--pre-js ./tfjs-backend-wasm/src/cc/pre.js",
    "--post-js ./tfjs-backend-wasm/src/cc/post.js",
]

# This build rule generates tfjs-backend-wasm.{js,wasm}.
#
# The ".js" at the end of the build name is significant because it determines
# the output file types. See the "-o <target>" flag in
# https://emscripten.org/docs/tools_reference/emcc.html
#
# Note that tfjs-backend-wasm.js will be used to load tfjs-backend-wasm.wasm
# as well as tfjs-backend-wasm-simd.wasm (the output of the next build rule).
cc_binary(
    name = "tfjs-backend-wasm.js",
    srcs = ["backend.cc"] + KERNELS_WITH_KEEPALIVE,
    linkopts = BASE_LINKOPTS + [
        "-s EXPORT_NAME=WasmBackendModule",
    ],
    deps = [
        ":all_kernels",
        ":backend",
    ],
)

cc_binary(
    name = "tfjs-backend-wasm-simd.js",
    srcs = ["backend.cc"] + KERNELS_WITH_KEEPALIVE,
    linkopts = BASE_LINKOPTS,
    deps = [
        ":all_kernels",
        ":backend",
    ],
)

cc_binary(
    name = "tfjs-backend-wasm-threaded-simd.js",
    srcs = ["backend.cc"] + KERNELS_WITH_KEEPALIVE,
    linkopts = BASE_LINKOPTS + [
        "-s EXPORT_NAME=WasmBackendModuleThreadedSimd",
        "-s MALLOC=emmalloc",
        "-s USE_PTHREADS=1",
        "-s PROXY_TO_PTHREAD=1",
        # Pre-create 8 webworkers (threads). The actual number of threads that
        # will be used by XNNPACK for creating the threadpool might be fewer. It
        # is by default set to the number of logical cores which in most cases
        # should be fewer than 8. It can also be set manually by calling the
        # `setThreadsCount` API.
        "-s PTHREAD_POOL_SIZE=8",
    ],
    deps = [
        ":all_kernels",
        ":backend",
    ],
)

wasm_cc_binary(
    name = "tfjs-backend-wasm",
    cc_target = ":tfjs-backend-wasm.js",
)

wasm_cc_binary(
    name = "tfjs-backend-wasm-simd",
    cc_target = ":tfjs-backend-wasm-simd.js",
    simd = True,
)

wasm_cc_binary(
    name = "tfjs-backend-wasm-threaded-simd",
    cc_target = ":tfjs-backend-wasm-threaded-simd.js",
    simd = True,
    threads = "emscripten",
)

test_suite(
    name = "cc_tests",
)

## Library

tfjs_cc_library(
    name = "backend",
    srcs = ["backend.cc"],
    hdrs = ["backend.h"],
    deps = [
        ":check_macros",
        ":util",
        "@xnnpack//:xnnpack_operators_nhwc_f32",
    ],
)

tfjs_unit_test(
    name = "backend_tests",
    srcs = glob(["*_test.cc"]),
    tags = ["ci"],
    deps = [
        ":Prelu",
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "binary",
    srcs = ["binary.cc"],
    hdrs = ["binary.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "check_macros",
    srcs = ["check_macros.h"],
    deps = [
        ":util",
    ],
)

tfjs_cc_library(
    name = "sin_cos_workaround",
    srcs = ["sin_cos_workaround.cc"],
    hdrs = ["sin_cos_workaround.h"],
)

tfjs_cc_library(
    name = "clamp_impl",
    srcs = ["clamp_impl.cc"],
    hdrs = ["clamp_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "conv2d_impl",
    srcs = ["conv2d_impl.cc"],
    hdrs = ["conv2d_impl.h"],
    deps = [
        ":backend",
        ":elu_impl",
        ":leakyrelu_impl",
        ":prelu_impl",
        ":sigmoid_impl",
        ":transpose_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "batch_mat_mul_impl",
    srcs = ["batch_mat_mul_impl.cc"],
    hdrs = ["batch_mat_mul_impl.h"],
    deps = [
        ":backend",
        ":elu_impl",
        ":leakyrelu_impl",
        ":prelu_impl",
        ":sigmoid_impl",
        ":transpose_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "elu_impl",
    srcs = ["elu_impl.cc"],
    hdrs = ["elu_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "interpolate_bilinear_impl",
    srcs = ["interpolate_bilinear_impl.cc"],
    hdrs = ["interpolate_bilinear_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "leakyrelu_impl",
    srcs = ["leakyrelu_impl.cc"],
    hdrs = ["leakyrelu_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "non_max_suppression_impl",
    srcs = ["non_max_suppression_impl.cc"],
    hdrs = ["non_max_suppression_impl.h"],
    deps = [
        ":backend",
    ],
)

tfjs_cc_library(
    name = "prelu_impl",
    srcs = ["prelu_impl.cc"],
    hdrs = ["prelu_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "sigmoid_impl",
    srcs = ["sigmoid_impl.cc"],
    hdrs = ["sigmoid_impl.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "transpose_impl",
    srcs = ["transpose_impl.cc"],
    hdrs = ["transpose_impl.h"],
    deps = [":util"],
)

tfjs_cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
)

tfjs_cc_library(
    name = "unary",
    srcs = ["unary.cc"],
    hdrs = ["unary.h"],
    deps = [
        ":backend",
    ],
)

# Kernels

tfjs_cc_library(
    name = "all_kernels",
    deps = [
        ":Abs",
        ":Add",
        ":AddN",
        ":All",
        ":Any",
        ":ArgMax",
        ":AvgPool",
        ":BatchMatMul",
        ":Ceil",
        ":ClipByValue",
        ":Conv2D",
        ":Conv2DBackpropInput",
        ":Cos",
        ":Cosh",
        ":CropAndResize",
        ":Cumsum",
        ":DepthToSpace",
        ":DepthwiseConv2dNative",
        ":Elu",
        ":Equal",
        ":Exp",
        ":FlipLeftRight",
        ":FloorDiv",
        ":FusedBatchNorm",
        ":FusedConv2D",
        ":FusedDepthwiseConv2D",
        ":Gather",
        ":GatherNd",
        ":Greater",
        ":GreaterEqual",
        ":LeakyRelu",
        ":Less",
        ":LessEqual",
        ":Max",
        ":MaxPool",
        ":Maximum",
        ":Min",
        ":Minimum",
        ":MirrorPad",
        ":Multiply",
        ":Neg",
        ":NonMaxSuppressionV3",
        ":NonMaxSuppressionV4",
        ":NonMaxSuppressionV5",
        ":NotEqual",
        ":OneHot",
        ":PadV2",
        ":Pow",
        ":Prelu",
        ":RealDiv",
        ":Relu",
        ":Relu6",
        ":ResizeBilinear",
        ":Reverse",
        ":RotateWithOffset",
        ":Round",
        ":ScatterNd",
        ":SelectV2",
        ":Sigmoid",
        ":Sin",
        ":Softmax",
        ":SparseFillEmptyRows",
        ":SparseReshape",
        ":SparseSegmentReduction",
        ":Square",
        ":SquaredDifference",
        ":StridedSlice",
        ":Sub",
        ":Tan",
        ":Tile",
        ":Transform",
        ":Transpose",
        ":_FusedMatMul",
    ],
)

tfjs_cc_library(
    name = "Abs",
    srcs = ["kernels/Abs.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
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
    name = "All",
    srcs = ["kernels/All.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Any",
    srcs = ["kernels/Any.cc"],
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
    name = "AvgPool",
    srcs = ["kernels/AvgPool.cc"],
    hdrs = ["kernels/AvgPool.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_unit_test(
    name = "AvgPool_test",
    srcs = ["kernels/AvgPool_test.cc"],
    tags = ["ci"],
    deps = [
        ":AvgPool",
    ],
)

tfjs_cc_library(
    name = "BatchMatMul",
    srcs = ["kernels/BatchMatMul.cc"],
    hdrs = ["kernels/BatchMatMul.h"],
    deps = [
        ":backend",
        ":batch_mat_mul_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "_FusedMatMul",
    srcs = ["kernels/_FusedMatMul.cc"],
    hdrs = ["kernels/_FusedMatMul.h"],
    deps = [
        ":backend",
        ":batch_mat_mul_impl",
    ],
)

tfjs_unit_test(
    name = "BatchMatMul_test",
    srcs = ["kernels/BatchMatMul_test.cc"],
    tags = ["ci"],
    deps = [
        ":BatchMatMul",
    ],
)

tfjs_unit_test(
    name = "_FusedMatMul_test",
    srcs = ["kernels/_FusedMatMul_test.cc"],
    tags = ["ci"],
    deps = [
        ":_FusedMatMul",
    ],
)

tfjs_cc_library(
    name = "Ceil",
    srcs = ["kernels/Ceil.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "ClipByValue",
    srcs = ["kernels/ClipByValue.cc"],
    hdrs = ["kernels/ClipByValue.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_unit_test(
    name = "ClipByValue_test",
    srcs = ["kernels/ClipByValue_test.cc"],
    tags = ["ci"],
    deps = [
        ":ClipByValue",
    ],
)

tfjs_cc_library(
    name = "Conv2D",
    srcs = ["kernels/Conv2D.cc"],
    hdrs = ["kernels/Conv2D.h"],
    deps = [
        ":backend",
        ":conv2d_impl",
    ],
)

tfjs_unit_test(
    name = "Conv2D_test",
    srcs = ["kernels/Conv2D_test.cc"],
    tags = ["ci"],
    deps = [
        ":Conv2D",
    ],
)

tfjs_cc_library(
    name = "Conv2DBackpropInput",
    srcs = ["kernels/Conv2DBackpropInput.cc"],
    deps = [
        ":backend",
    ],
)

tfjs_cc_library(
    name = "Cos",
    srcs = ["kernels/Cos.cc"],
    deps = [
        ":backend",
        ":sin_cos_workaround",
        ":unary",
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
    name = "Cumsum",
    srcs = ["kernels/Cumsum.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "DepthToSpace",
    srcs = ["kernels/DepthToSpace.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "DepthwiseConv2dNative",
    srcs = ["kernels/DepthwiseConv2dNative.cc"],
    hdrs = ["kernels/DepthwiseConv2dNative.h"],
    deps = [
        ":backend",
        ":conv2d_impl",
    ],
)

tfjs_unit_test(
    name = "DepthwiseConv2dNative_test",
    srcs = ["kernels/DepthwiseConv2dNative_test.cc"],
    tags = ["ci"],
    deps = [
        ":DepthwiseConv2dNative",
    ],
)

tfjs_cc_library(
    name = "RealDiv",
    srcs = ["kernels/RealDiv.cc"],
    deps = [
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Elu",
    srcs = ["kernels/Elu.cc"],
    hdrs = ["kernels/Elu.h"],
    deps = [
        ":backend",
        ":elu_impl",
        ":unary",
    ],
)

tfjs_cc_library(
    name = "Equal",
    srcs = ["kernels/Equal.cc"],
    deps = [
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Exp",
    srcs = ["kernels/Exp.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "FlipLeftRight",
    srcs = ["kernels/FlipLeftRight.cc"],
    deps = [
        ":backend",
        ":util",
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
    name = "FusedBatchNorm",
    srcs = ["kernels/FusedBatchNorm.cc"],
    deps = [
        ":backend",
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

tfjs_unit_test(
    name = "FusedConv2D_test",
    srcs = ["kernels/FusedConv2D_test.cc"],
    tags = ["ci"],
    deps = [
        ":FusedConv2D",
    ],
)

tfjs_cc_library(
    name = "FusedDepthwiseConv2D",
    srcs = ["kernels/FusedDepthwiseConv2D.cc"],
    hdrs = ["kernels/FusedDepthwiseConv2D.h"],
    deps = [
        ":backend",
        ":conv2d_impl",
    ],
)

tfjs_unit_test(
    name = "FusedDepthwiseConv2D_test",
    srcs = ["kernels/FusedDepthwiseConv2D_test.cc"],
    tags = ["ci"],
    deps = [
        ":FusedDepthwiseConv2D",
    ],
)

tfjs_cc_library(
    name = "Gather",
    srcs = ["kernels/Gather.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "GatherNd",
    srcs = ["kernels/GatherNd.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Greater",
    srcs = ["kernels/Greater.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "GreaterEqual",
    srcs = ["kernels/GreaterEqual.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "LeakyRelu",
    srcs = ["kernels/LeakyRelu.cc"],
    hdrs = ["kernels/LeakyRelu.h"],
    deps = [
        ":backend",
        ":leakyrelu_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Less",
    srcs = ["kernels/Less.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "LessEqual",
    srcs = ["kernels/LessEqual.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "NotEqual",
    srcs = ["kernels/NotEqual.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "LogicalAnd",
    srcs = ["kernels/LogicalAnd.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Log",
    srcs = ["kernels/Log.cc"],
    deps = [
        ":backend",
        ":unary",
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
    name = "Maximum",
    srcs = ["kernels/Maximum.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "MaxPool",
    srcs = ["kernels/MaxPool.cc"],
    hdrs = ["kernels/MaxPool.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_unit_test(
    name = "MaxPool_test",
    srcs = ["kernels/MaxPool_test.cc"],
    tags = ["ci"],
    deps = [
        ":MaxPool",
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
    name = "Minimum",
    srcs = ["kernels/Minimum.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "MirrorPad",
    srcs = ["kernels/MirrorPad.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Multiply",
    srcs = ["kernels/Multiply.cc"],
    deps = [
        ":backend",
        ":binary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Neg",
    srcs = ["kernels/Neg.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "NonMaxSuppressionV3",
    srcs = ["kernels/NonMaxSuppressionV3.cc"],
    deps = [
        ":backend",
        ":non_max_suppression_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "NonMaxSuppressionV4",
    srcs = ["kernels/NonMaxSuppressionV4.cc"],
    deps = [
        ":backend",
        ":non_max_suppression_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "NonMaxSuppressionV5",
    srcs = ["kernels/NonMaxSuppressionV5.cc"],
    deps = [
        ":backend",
        ":non_max_suppression_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "OneHot",
    srcs = ["kernels/OneHot.cc"],
    deps = [
        ":backend",
    ],
)

tfjs_cc_library(
    name = "PadV2",
    srcs = ["kernels/PadV2.cc"],
    hdrs = ["kernels/PadV2.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_unit_test(
    name = "PadV2_test",
    srcs = ["kernels/PadV2_test.cc"],
    tags = ["ci"],
    deps = [
        ":PadV2",
    ],
)

tfjs_cc_library(
    name = "Pow",
    srcs = ["kernels/Pow.cc"],
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
        ":prelu_impl",
        ":util",
    ],
)

tfjs_unit_test(
    name = "Prelu_test",
    srcs = ["kernels/Prelu_test.cc"],
    tags = ["ci"],
    deps = [
        ":Prelu",
    ],
)

tfjs_cc_library(
    name = "Relu",
    srcs = ["kernels/Relu.cc"],
    deps = [
        ":backend",
        ":clamp_impl",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Relu6",
    srcs = ["kernels/Relu6.cc"],
    deps = [
        ":backend",
        ":clamp_impl",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "ResizeBilinear",
    srcs = ["kernels/ResizeBilinear.cc"],
    hdrs = ["kernels/ResizeBilinear.h"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_unit_test(
    name = "ResizeBilinear_test",
    srcs = ["kernels/ResizeBilinear_test.cc"],
    tags = ["ci"],
    deps = [
        ":ResizeBilinear",
    ],
)

tfjs_cc_library(
    name = "Reverse",
    srcs = ["kernels/Reverse.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "RotateWithOffset",
    srcs = ["kernels/RotateWithOffset.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Round",
    srcs = ["kernels/Round.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "ScatterNd",
    srcs = ["kernels/ScatterNd.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "SelectV2",
    srcs = ["kernels/SelectV2.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Sigmoid",
    srcs = ["kernels/Sigmoid.cc"],
    hdrs = ["kernels/Sigmoid.h"],
    deps = [
        ":backend",
        ":sigmoid_impl",
        ":unary",
    ],
)

tfjs_cc_library(
    name = "Sin",
    srcs = ["kernels/Sin.cc"],
    deps = [
        ":backend",
        ":sin_cos_workaround",
        ":unary",
    ],
)

tfjs_cc_library(
    name = "Softmax",
    srcs = ["kernels/Softmax.cc"],
    hdrs = ["kernels/Softmax.h"],
    deps = [
        ":backend",
        ":unary",
    ],
)

tfjs_unit_test(
    name = "Sigmoid_test",
    srcs = ["kernels/Sigmoid_test.cc"],
    tags = ["ci"],
    deps = [
        ":Sigmoid",
    ],
)

tfjs_unit_test(
    name = "Softmax_test",
    srcs = ["kernels/Softmax_test.cc"],
    tags = ["ci"],
    deps = [
        ":Softmax",
    ],
)

tfjs_cc_library(
    name = "SparseFillEmptyRows",
    srcs = ["kernels/SparseFillEmptyRows.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "SparseReshape",
    srcs = ["kernels/SparseReshape.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "SparseSegmentReduction",
    srcs = ["kernels/SparseSegmentReduction.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Square",
    srcs = ["kernels/Square.cc"],
    deps = [
        ":backend",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "SquaredDifference",
    srcs = ["kernels/SquaredDifference.cc"],
    deps = [
        ":backend",
        ":binary",
        ":unary",
        ":util",
    ],
)

tfjs_cc_library(
    name = "StridedSlice",
    srcs = ["kernels/StridedSlice.cc"],
    deps = [
        ":backend",
        ":util",
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
    name = "Tile",
    srcs = ["kernels/Tile.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "TopK",
    srcs = ["kernels/TopK.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Transform",
    srcs = ["kernels/Transform.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Transpose",
    srcs = ["kernels/Transpose.cc"],
    deps = [
        ":backend",
        ":transpose_impl",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Step",
    srcs = ["kernels/Step.cc"],
    deps = [
        ":backend",
        ":util",
    ],
)

tfjs_cc_library(
    name = "Tan",
    srcs = ["kernels/Tan.cc"],
    deps = [
        ":backend",
        ":sin_cos_workaround",
        ":unary",
    ],
)

tfjs_cc_library(
    name = "Cosh",
    srcs = ["kernels/Cosh.cc"],
    deps = [
        ":backend",
        ":unary",
    ],
)
