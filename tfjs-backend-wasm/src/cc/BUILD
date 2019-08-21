cc_binary(
    name = "tfjs-backend-wasm.js",
    srcs = ["lib.cc"],
    # TODO(smilkov): Make sure these flags are passed when making an
    # optimized build: -g0 -O3 --llvm-lto 3
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
    deps = [":kernels", ":util"],
)

cc_library(
  name = "kernels",
  hdrs = ["kernels.h"],
  srcs = ["kernels.cc"],
)

cc_library(
  name = "util",
  srcs = ["util.h"],
)
