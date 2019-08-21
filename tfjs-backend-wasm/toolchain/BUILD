package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

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
        "emar.sh",
        "false.sh",
        "@emsdk//:all",
    ],
)

cc_toolchain(
   name = "asmjs_toolchain",
   toolchain_identifier = "asmjs-toolchain",
   toolchain_config = ":asmjs_toolchain_config",
   all_files = ":all",
   compiler_files = ":all",
   dwp_files = ":empty",
   linker_files = ":all",
   objcopy_files = ":empty",
   strip_files = ":empty",
   supports_param_files = 0,
)
