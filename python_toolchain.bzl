load("@rules_python//python:defs.bzl", "py_runtime_pair")


def configure_python_toolchain(name, platform_data):
    
    py2_runtime_name = "python2_runtime_%s" % name
    py3_runtime_name = "python3_runtime_%s" % name
    runtime_pair_name = "tfjs_py_runtime_pair_%s" % name

    native.py_runtime(
        name = py3_runtime_name,
        files = ["@python3_interpreter_%s//:files" % name],
        interpreter = "@python3_interpreter_%s//:python3_bin" % name,
        python_version = "PY3",
        visibility = ["//visibility:public"],
    )

    native.py_runtime(
        name = py2_runtime_name,
        files = ["@python2_interpreter_%s//:files" % name],
        interpreter = "@python2_interpreter_%s//:python_bin" % name,
        python_version = "PY2",
        visibility = ["//visibility:public"],
    )

    py_runtime_pair(
        name = runtime_pair_name,
        py2_runtime = ":%s" % py2_runtime_name,
        py3_runtime = ":%s" % py3_runtime_name,
    )

    native.toolchain(
        name = "tfjs_py_toolchain_%s" % name,
        toolchain = ":%s" % runtime_pair_name,
        toolchain_type = "@bazel_tools//tools/python:toolchain_type",
        exec_compatible_with = platform_data.exec_compatible_with,
    )



def configure_python_toolchains(platforms = {}):

    for name, platform_data in platforms.items():
        configure_python_toolchain(
            name = name,
            platform_data = platform_data,
        )

    # native.alias(
    #     name = "tensorflowjs_dev_deps",
    #     actual = select({
    #         "@platforms//os:macos": "@tensorflowjs_dev_deps_darwin_amd64//:requirements.bzl", "requirement",
    #         "@platforms//os:linux": "@python_interpreter_linux_amd64//:python_bin",
    #         "@platforms//os:windows": "@python_interpreter_windows_amd64//:python_bin",
    #     })
    # )
            
# def dev_requirement(name):
#     alias(
#         name = "tensorflowjs_dev_deps_%s" % name,
#         actual = select({
#             "@platforms//os:macos": "@tensorflowjs_dev_deps_darwin_amd64//:requirements.bzl", "requirement",
#             "@platforms//os:linux": "@python_interpreter_linux_amd64//:python_bin",
#             "@platforms//os:windows": "@python_interpreter_windows_amd64//:python_bin",
#         })

