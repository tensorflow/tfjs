# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

load("@bazel_skylib//lib:paths.bzl", "paths")

def _copy_to_dist_impl(ctx):
    files = [f for s in ctx.attr.srcs for f in s.files.to_list()]

    root_dir = paths.join(paths.dirname(ctx.build_file_path), ctx.attr.root)

    outputs = []
    for f in files:
        dest_path = paths.join(ctx.attr.dest_dir, paths.relativize(f.short_path, root_dir))
        out = ctx.actions.declare_file(dest_path)
        outputs.append(out)
        ctx.actions.symlink(
            output = out,
            target_file = f,
        )

    return [DefaultInfo(files = depset(outputs))]

copy_to_dist = rule(
    implementation = _copy_to_dist_impl,
    attrs = {
        "dest_dir": attr.string(
            default = "dist",
            doc = "Destination directory to copy the source file tree to",
        ),
        "root": attr.string(
            default = "",
            doc = "Common root path to remove when symlinking. Relative to the build file's directory",
        ),
        "srcs": attr.label_list(
            doc = "Files to create symlinks of",
        ),
    },
    doc = """Creates symlinks in 'dest_dir' for each file in 'srcs'

    Preserves relative paths between linked files. A common root path of the
    input files can be removed via the 'root' arg.

    This rule is used to 'copy' the results of compiling a tfjs package's
    sources located in 'src' to the output directory 'dist' while preserving
    the filetree's relative paths. The benifit of using this rule is that Bazel
    is aware of the copied files and can use them in other rules like 'pkg_npm'.
    It is also used for copying tfjs bundles and miniprogram outputs.
    """,
)
