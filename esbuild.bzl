"""Automatic esbuild binary selection based on platform"""

load("@npm//@bazel/esbuild:index.bzl", _esbuild = "esbuild")

def esbuild(**kwargs):
    _esbuild(
        tool = select({
            "@bazel_tools//src/conditions:darwin": "@esbuild_darwin//:bin/esbuild",
            "@bazel_tools//src/conditions:linux_x86_64": "@esbuild_linux//:bin/esbuild",
            "@bazel_tools//src/conditions:windows": "@esbuild_windows//:esbuild.exe",
        }),
        **kwargs
    )
