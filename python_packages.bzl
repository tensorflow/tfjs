_py2_from_source_build_file_content = """
exports_files(["python_bin"])
filegroup(
    name = "files",
    srcs = glob(["bazel_install_py2/**"], exclude = ["**/* *"]),
    visibility = ["//visibility:public"],
)
"""

_py2_configure = """
./configure --prefix=$(pwd)/bazel_install_py2
"""

_py2_configure_darwin = """
./configure --prefix=$(pwd)/bazel_install_py2 --with-openssl=$(brew --prefix openssl)
"""

def _py2_patch_cmds(configure):
    return [
        "mkdir $(pwd)/bazel_install_py2",
        configure,
        "make -j",
        "make install",
        "ln -s bazel_install_py2/bin/python python_bin",
    ]


_py3_from_source_build_file_content = """
exports_files(["python3_bin"])
filegroup(
    name = "files",
    srcs = glob(["bazel_install_py3/**"], exclude = ["**/* *"]),
    visibility = ["//visibility:public"],
)
"""

_py3_configure = """
./configure --prefix=$(pwd)/bazel_install_py3
"""

_py3_configure_darwin = """
./configure --prefix=$(pwd)/bazel_install_py3 --with-openssl=$(brew --prefix openssl)
"""

def _py3_patch_cmds(configure):
    return [
        "mkdir $(pwd)/bazel_install_py3",
        configure,
        "make -j",
        "make install",
        "ln -s bazel_install_py3/bin/python3 python3_bin",
    ]

PYTHON_PACKAGES = dict({
    "darwin_amd64": struct(
        python2 = struct(
            sha256 = "da3080e3b488f648a3d7a4560ddee895284c3380b11d6de75edb986526b9a814",
            urls = [
                "https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz",
            ],
            build_file_content = _py2_from_source_build_file_content,
            patch_cmds = _py2_patch_cmds(_py2_configure_darwin),
            strip_prefix = "Python-2.7.18",
        ),
        python3 = struct(
            sha256 = "fb1a1114ebfe9e97199603c6083e20b236a0e007a2c51f29283ffb50c1420fb2",
            urls = [
                "https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tar.xz",
            ],
            build_file_content = _py3_from_source_build_file_content,
            patch_cmds = _py3_patch_cmds(_py3_configure_darwin),
            strip_prefix = "Python-3.8.11",
        ),
        exec_compatible_with = [
            "@platforms//os:macos",
            "@platforms//cpu:x86_64",
        ]
    ),
    "linux_amd64": struct(
        python2 = struct(
            sha256 = "da3080e3b488f648a3d7a4560ddee895284c3380b11d6de75edb986526b9a814",
            urls = [
                "https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz",
            ],
            build_file_content = _py2_from_source_build_file_content,
            patch_cmds = _py2_patch_cmds(_py2_configure),
            strip_prefix = "Python-2.7.18",
        ),
        python3 = struct(
            sha256 = "fb1a1114ebfe9e97199603c6083e20b236a0e007a2c51f29283ffb50c1420fb2",
            urls = [
                "https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tar.xz",
            ],
            build_file_content = _py3_from_source_build_file_content,
            patch_cmds = _py3_patch_cmds(_py3_configure),
            strip_prefix = "Python-3.8.11",
        ),
        exec_compatible_with = [
            "@platforms//os:linux",
            "@platforms//cpu:x86_64",
        ]
    ),
    "windows_amd64": struct(
        python2 = struct(
            sha256 = "da3080e3b488f648a3d7a4560ddee895284c3380b11d6de75edb986526b9a814",
            urls = [
                "https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz",
            ],
            build_file_content = _py2_from_source_build_file_content,
            patch_cmds = _py2_patch_cmds(_py2_configure),
            strip_prefix = "Python-2.7.18",
        ),
        python3 = struct(
            sha256 = "fb1a1114ebfe9e97199603c6083e20b236a0e007a2c51f29283ffb50c1420fb2",
            urls = [
                "https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tar.xz",
            ],
            build_file_content = _py3_from_source_build_file_content,
            patch_cmds = _py3_patch_cmds(_py3_configure),
            strip_prefix = "Python-3.8.11",
        ),
        exec_compatible_with = [
            "@platforms//os:windows",
            "@platforms//cpu:x86_64",
        ]
    ),
})
