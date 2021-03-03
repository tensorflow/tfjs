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
                            # The clang compiler comes with a definition of
                            # max_align_t struct in $emsdk/upstream/lib/clang/13.0.0/include/__stddef_max_align_t.h.
                            # It conflicts with the one defined in
                            # $emsdk/upstream/emscripten/cache/sysroot/include/bits/alltypes.h.
                            # We need both include paths to make things work.
                            #
                            # To workaround this, we are defining the following
                            # symbol through compiler flag so that the max_align_t
                            # defined in clang's header file will be skipped.
                            "-D",
                            "__CLANG_MAX_ALIGN_T_DEFINED",
                            # We are using emscripten 2.0.14 for this build. It
                            # comes with clang 13.0.0. Future emscripten release
                            # might change the clang version number below.
                            #
                            # Also need to change the version number in
                            # cxx_cxx_builtin_include_directories below.
                            "-isystem",
                            "external/emsdk/emsdk/upstream/lib/clang/13.0.0/include",
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

    cxx_builtin_include_directories = [
        "external/emsdk/emsdk/upstream/emscripten/cache/sysroot/include/c++/v1",
        "external/emsdk/emsdk/upstream/emscripten/cache/sysroot/include/compat",
        "external/emsdk/emsdk/upstream/emscripten/cache/sysroot/include",
        # We are using emscripten 2.0.14 for this build. It comes with clang
        # 13.0.0. Future emscripten release might change the clang version
        # number below.
        #
        # Also need to change the version number in
        # toolchain_include_directories_feature above.
        "external/emsdk/emsdk/upstream/lib/clang/13.0.0/include",
    ]

    builtin_sysroot = "external/emsdk/emsdk/upstream/emscripten/cache/sysroot"

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "wasm-toolchain",
        host_system_name = "i686-unknown-linux-gnu",
        target_system_name = "wasm-unknown-emscripten",
        target_cpu = "wasm",
        target_libc = "musl/js",
        compiler = "emscripten",
        abi_version = "emscripten_syscalls",
        abi_libc_version = "default",
        tool_paths = tool_paths,
        features = features,
        builtin_sysroot = builtin_sysroot,
        cxx_builtin_include_directories = cxx_builtin_include_directories,
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
