"""Re-export of some bazel rules with repository-wide defaults."""

load("@npm//@bazel/esbuild:index.bzl", _esbuild = "esbuild")
load("@npm//@bazel/typescript:index.bzl", _ts_library = "ts_library")

def ts_library(**kwargs):
    # Use the ts_library tsconfig by default. The ts_library tsconfig does not
    # have "incremental = true" because ts_library does not support incremental
    # builds at the level of the typescript compiler (it supports incrementally
    # rebuilding at the level of Bazel build targets).
    tsconfig = kwargs.pop("tsconfig", "@//:tsconfig_ts_library.json")

    # Use es2017 by default. This is transpiled down to es5 in tfjs_bundle.bzl.
    devmode_target = kwargs.pop("devmode_target", "es2017")
    prodmode_target = kwargs.pop("prodmode_target", "es2017")

    _ts_library(
        tsconfig = tsconfig,
        devmode_target = devmode_target,
        prodmode_target = prodmode_target,
        **kwargs
    )

def esbuild(**kwargs):
    _esbuild(
        tool = select({
            "@bazel_tools//src/conditions:darwin": "@esbuild_darwin//:bin/esbuild",
            "@bazel_tools//src/conditions:linux_x86_64": "@esbuild_linux//:bin/esbuild",
            "@bazel_tools//src/conditions:windows": "@esbuild_windows//:esbuild.exe",
        }),
        **kwargs
    )
