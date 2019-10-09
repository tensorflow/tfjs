cc_binary(
    name = "tfjs-backend-wasm.js",
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
        ":backend",
        ":all_kernels",
    ],
)

cc_library(
  name = "backend",
  srcs = ["backend.cc"],
  hdrs = ["backend.h"],
  linkstatic = 1,
  deps = [
    ":util",
    "@xnnpack//:XNNPACK",
  ],
)

cc_library(
  name = "all_kernels",
  linkstatic = 1,
  deps = [
    ":add",
    ":batch_matmul",
  ]
)

cc_library(
  name = "add",
  srcs = ["add.cc"],
  linkstatic = 1,
  deps = [
    ":backend",
    ":util",
  ],
)

cc_library(
  name = "batch_matmul",
  srcs = ["batch_matmul.cc"],
  linkstatic = 1,
  deps = [
    ":backend",
    ":util",
    "@xnnpack//:XNNPACK",
  ],
)

cc_library(
  name = "util",
  linkstatic = 1,
  srcs = ["util.h"],
)
