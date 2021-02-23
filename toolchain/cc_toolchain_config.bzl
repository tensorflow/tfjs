load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
    "with_feature_set",
)

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
    preprocessor_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.clif_match,
    ]

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
        ACTION_NAMES.lto_backend,
    ]
    toolchain_include_directories_feature = feature(
        name = "toolchain_include_directories",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_compile_actions,
                flag_groups = [
                    flag_group(
                        flags = [
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/include/libcxx",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/lib/libcxxabi/include",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/include/compat",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/include",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/include/libc",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/lib/libc/musl/arch/emscripten",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/emscripten/system/local/include",
                            "-isystem",
                            "external/emsdk/emsdk/upstream/lib/clang/11.0.0/include",
                        ],
                    ),
                ],
            ),
        ],
    )

    crosstool_default_flag_sets = [
        # Opt.
        flag_set(
            actions = preprocessor_compile_actions,
            flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            with_features = [with_feature_set(features = ["opt"])],
        ),
        flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [flag_group(flags = ["-g0", "-O3"])],
            with_features = [with_feature_set(features = ["opt"])],
        ),
        # Fastbuild.
        flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [flag_group(flags = ["-O2"])],
            with_features = [with_feature_set(features = ["fastbuild"])],
        ),
        # Dbg.
        flag_set(
            actions = all_compile_actions + all_link_actions,
            flag_groups = [flag_group(flags = ["-g2", "-O0"])],
            with_features = [with_feature_set(features = ["dbg"])],
        ),
    ]

    features = [
        toolchain_include_directories_feature,
        # These 3 features will be automatically enabled by blaze in the
        # corresponding build mode.
        feature(
            name = "opt",
            provides = ["variant:crosstool_build_mode"],
        ),
        feature(
            name = "dbg",
            provides = ["variant:crosstool_build_mode"],
        ),
        feature(
            name = "fastbuild",
            provides = ["variant:crosstool_build_mode"],
        ),
        feature(
            name = "crosstool_default_flags",
            enabled = True,
            flag_sets = crosstool_default_flag_sets,
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "wasm-toolchain",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "wasm-unknown-emscripten",
        target_cpu = "wasm",
        target_libc = "unknown",
        compiler = "emscripten",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
        features = features,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)

def _emsdk_impl(ctx):
    if "EMSDK" not in ctx.os.environ or ctx.os.environ["EMSDK"].strip() == "":
        fail("The environment variable EMSDK is not found. " +
             "Did you run source ./emsdk_env.sh ?")
    path = ctx.os.environ["EMSDK"]
    ctx.symlink(path, "emsdk")
    ctx.file("BUILD", """
filegroup(
    name = "all",
    srcs = glob(["emsdk/**"]),
    visibility = ["//visibility:public"],
)
""")

emsdk_configure = repository_rule(
    implementation = _emsdk_impl,
    local = True,
)
