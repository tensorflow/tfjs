package(default_visibility = ["//visibility:public"])

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

cc_toolchain_config(name = "wasm_toolchain_config")

cc_toolchain_suite(
    name = "emscripten",
    toolchains = {
        "wasm": ":wasm_toolchain",
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
   name = "wasm_toolchain",
   toolchain_identifier = "wasm-toolchain",
   toolchain_config = ":wasm_toolchain_config",
   all_files = ":all",
   compiler_files = ":all",
   dwp_files = ":empty",
   linker_files = ":all",
   objcopy_files = ":empty",
   strip_files = ":empty",
   supports_param_files = 0,
)
