cc_binary(
    name = "wasm.js",
    srcs = ["lib.cc"],
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
