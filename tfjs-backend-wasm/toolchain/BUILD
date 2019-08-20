package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

filegroup(
  name = "emscripten_cache_content",
  srcs = glob(["emscripten_cache/**/*"]),
)

cc_toolchain_config(name = "asmjs_toolchain_config")

cc_toolchain_suite(
    name = "emscripten",
    toolchains = {
        "asmjs": ":asmjs_toolchain",
    },
)

filegroup(
    name = "all",
    srcs = [
        "emcc.sh",
        "@emscripten_clang//:all",
        "@emscripten_toolchain//:all",
        ":emscripten_cache_content",
    ],
)

cc_toolchain(
   name = "asmjs_toolchain",
   toolchain_identifier = "asmjs-toolchain",
   toolchain_config = ":asmjs_toolchain_config",
   all_files = ":all",
   compiler_files = ":all",
   cpu = "asmjs",
   dwp_files = ":empty",
   linker_files = ":all",
   objcopy_files = ":empty",
   strip_files = ":empty",
   supports_param_files = 0,
)
