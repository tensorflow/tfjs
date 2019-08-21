load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
     "feature",
     "flag_group",
     "flag_set",
     "tool_path")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "emcc.sh",
        ),
        tool_path(
            name = "ld",
            path = "emcc.sh",
        ),
        tool_path(
            name = "ar",
            path = "emar.sh",
        ),
        tool_path(
            name = "cpp",
            path = "false.sh",
        ),
        tool_path(
            name = "gcov",
            path = "false.sh",
        ),
        tool_path(
            name = "nm",
            path = "NOT_USED",
        ),
        tool_path(
            name = "objdump",
            path = "false.sh",
        ),
        tool_path(
            name = "strip",
            path = "NOT_USED",
        ),
    ]
    toolchain_include_directories_feature = feature(
        name = "toolchain_include_directories",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                  ACTION_NAMES.assemble,
                  ACTION_NAMES.preprocess_assemble,
                  ACTION_NAMES.linkstamp_compile,
                  ACTION_NAMES.c_compile,
                  ACTION_NAMES.cpp_compile,
                  ACTION_NAMES.cpp_header_parsing,
                  ACTION_NAMES.cpp_module_compile,
                  ACTION_NAMES.cpp_module_codegen,
                  ACTION_NAMES.lto_backend,
                  ACTION_NAMES.clif_match,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-isystem",
                            "external/emsdk/emsdk/fastcomp/emscripten/system/include/libcxx",
                            "-isystem",
                            "external/emsdk/emsdk/fastcomp/emscripten/system/include/libc",
                            "-isystem",
                            "external/emsdk/emsdk/fastcomp/emscripten/system/include/emscripten",
                        ],
                    ),
                ],
            ),
        ],
    )

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "asmjs-toolchain",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "asmjs-unknown-emscripten",
        target_cpu = "asmjs",
        target_libc = "unknown",
        compiler = "emscripten",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        features = [toolchain_include_directories_feature],
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)

def _emsdk_impl(ctx):
  path = '%s/emsdk' % ctx.os.environ['HOME']
  ctx.symlink(path, "emsdk")
  ctx.file("BUILD", """
filegroup(
    name = "all",
    srcs = glob(["emsdk/**"]),
    visibility = ["//visibility:public"],
)
""")

emsdk_configure = repository_rule(
    implementation=_emsdk_impl,
    local = True)
