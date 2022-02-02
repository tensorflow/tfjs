load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_python//python:pip.bzl", "pip_install")
load(":python_packages.bzl", "PYTHON_PACKAGES")

def python_repositories():
    """Load and register python toolchains from PYTHON_PACKAGES.

    Declare `http_archive`s, register toolchains, and define `pip_install`
    external repositories for each platform in PYTHON_PACKAGES.
    """
    for name, platform_data in PYTHON_PACKAGES.items():
        if name not in native.existing_rules():
            py2 = platform_data.python2 if "python2" in dir(platform_data) else None
            py3 = platform_data.python3

            if py2:
                http_archive(
                    name = "python2_interpreter_%s" % name,
                    build_file_content = py2.build_file_content,
                    patch_cmds = py2.patch_cmds,
                    sha256 = py2.sha256,
                    strip_prefix = py2.strip_prefix,
                    urls = py2.urls,
                )

            http_archive(
                name = "python3_interpreter_%s" % name,
                build_file_content = py3.build_file_content,
                patch_cmds = py3.patch_cmds,
                sha256 = py3.sha256,
                strip_prefix = py3.strip_prefix,
                urls = py3.urls,
            )

            native.register_toolchains("@//:tfjs_py_toolchain_%s" % name)

            # Create a central external repo, @tensorflowjs_dev_deps, that
            # contains Bazel targets for all the third-party packages specified
            # in the requirements.txt file.
            interpreter = "@python3_interpreter_%s//:%s" % (name, py3.python_interpreter)
            pip_install(
                name = "tensorflowjs_dev_deps_%s" % name,
                python_interpreter_target = interpreter,
                requirements = "@//tfjs-converter/python:requirements-dev.txt",
            )

            pip_install(
                name = "tensorflowjs_deps_%s" % name,
                python_interpreter_target = interpreter,
                requirements = "@//tfjs-converter/python:requirements.txt",
            )

        # Create an external repo for python deps that relies on the default
        # python interpreter.
        pip_install(
            name = "tensorflowjs_dev_deps",
            requirements = "@//tfjs-converter/python:requirements-dev.txt",
        )

        pip_install(
            name = "tensorflowjs_deps",
            requirements = "@//tfjs-converter/python:requirements.txt",
        )
