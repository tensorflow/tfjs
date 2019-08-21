cc_binary(
    name = "tfjs-backend-wasm.js",
    srcs = ["lib.cc"],
    linkopts = [
        "-s ALLOW_MEMORY_GROWTH=1",
        "-s DEFAULT_LIBRARY_FUNCS_TO_INCLUDE=[]",
        "-s DISABLE_EXCEPTION_CATCHING=1",
        "-s FILESYSTEM=0",
        "-s EXIT_RUNTIME=0",
        "-s EXPORTED_FUNCTIONS='[\"_malloc\"]'",
        "-s EXTRA_EXPORTED_RUNTIME_METHODS='[\"cwrap\"]'",
        "-s ENVIRONMENT=web",
        "-s MODULARIZE=1",
        "-s EXPORT_NAME=WasmBackendModule",
        "-s MALLOC=emmalloc",
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
