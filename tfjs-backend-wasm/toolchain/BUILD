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
        "emar.sh",
        "emcc.sh",
        "false.sh",
        "@emsdk//:all",
    ],
)

cc_toolchain(
    name = "wasm_toolchain",
    all_files = ":all",
    compiler_files = ":all",
    dwp_files = ":empty",
    linker_files = ":all",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":wasm_toolchain_config",
    toolchain_identifier = "wasm-toolchain",
)
