load("@rules_python//python:defs.bzl", "py_runtime_pair")
load(":python_packages.bzl", "PYTHON_PACKAGES")

def configure_python_toolchain(name, platform_data):
    """Define and configure the python toolchain for a given platform.

    For a given platform, create a py_runtime for python2 and python3 and a
    corresponding py_runtime_pair. Define the toolchain and provide a
    config_setting for use with `select` in `dev_requirement`.

    The toolchain is registered in the `python_repositories` workspace function.
    """

    py2 = platform_data.python2 if "python2" in dir(platform_data) else None
    py3 = platform_data.python3

    py2_runtime_name = "python2_runtime_%s" % name
    py3_runtime_name = "python3_runtime_%s" % name
    runtime_pair_name = "tfjs_py_runtime_pair_%s" % name

    native.py_runtime(
        name = py3_runtime_name,
        files = ["@python3_interpreter_%s//:files" % name],
        interpreter = "@python3_interpreter_%s//:%s" % (name, py3.python_interpreter),
        python_version = "PY3",
        visibility = ["//visibility:public"],
    )

    if py2:
        native.py_runtime(
            name = py2_runtime_name,
            files = ["@python2_interpreter_%s//:files" % name],
            interpreter = "@python2_interpreter_%s//:%s" % (name, py2.python_interpreter),
            python_version = "PY2",
            visibility = ["//visibility:public"],
        )

    py_runtime_pair(
        name = runtime_pair_name,
        py2_runtime = ":%s" % py2_runtime_name if py2 else None,
        py3_runtime = ":%s" % py3_runtime_name,
    )

    native.toolchain(
        name = "tfjs_py_toolchain_%s" % name,
        toolchain = ":%s" % runtime_pair_name,
        toolchain_type = "@bazel_tools//tools/python:toolchain_type",
        exec_compatible_with = platform_data.exec_compatible_with,
    )

    native.config_setting(
        name = name,
        constraint_values = platform_data.exec_compatible_with,
    )

def configure_python_toolchains(platforms = PYTHON_PACKAGES):
    """Set up multiple python toolchains

    Configures a python toolchain for each platform.
    """
    for name, platform_data in platforms.items():
        configure_python_toolchain(
            name = name,
            platform_data = platform_data,
        )

def dev_requirement(name):
    """Get the dev target name corresponding to the python pip package name

    This reproduces functionalty of the `requirement` function exported by
    `requirements.bzl` in the python toolchain. It's necessary because there
    is no way to `select` the appropriate `requirement` function from a python
    platform without causing all the platforms to be downloaded and built.

    https://github.com/bazelbuild/rules_python/blob/main/python/pip_install/repositories.bzl#L60-L62
    """

    alias_name = "tensorflowjs_dev_deps_%s" % name
    if alias_name not in native.existing_rules():
        _name = name.replace("-", "_").lower()
        selection_values = {}
        for platform_name in PYTHON_PACKAGES.keys():
            selection_values["@//:%s" % platform_name] = \
                "@tensorflowjs_dev_deps_%s//pypi__%s" % (platform_name, _name)

        selection_values["//conditions:default"] = \
            "@tensorflowjs_dev_deps//pypi__%s" % _name

        native.alias(
            name = alias_name,
            actual = select(selection_values),
        )
    return alias_name
