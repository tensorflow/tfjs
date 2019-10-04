cc_binary(
    name = "tfjs-backend-wasm.js",
    srcs = ["lib.cc"],
    linkopts = [
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        "-s FILESYSTEM=0",
        "-s EXIT_RUNTIME=0",
        "-s EXPORTED_FUNCTIONS='[\"_malloc\", \"_free\"]'",
        "-s EXTRA_EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
        "-s ENVIRONMENT=web",
        "-s ASSERTIONS=1",
        "-s SAFE_HEAP=1",
        "-s SAFE_HEAP_LOG=1",
        "-s MODULARIZE=1",
        "-s STACK_OVERFLOW_CHECK=1",
        "-s TOTAL_MEMORY=512MB",
        "-s EXPORT_NAME=WasmBackendModule",
        "-s MALLOC=emmalloc",
        "-g",
    ],
    deps = [
        ":kernels",
        ":util",
    ],
)

cc_library(
    name = "kernels",
    srcs = ["kernels.cc"],
    hdrs = ["kernels.h"],
)

cc_library(
    name = "util",
    srcs = ["util.h"],
)
