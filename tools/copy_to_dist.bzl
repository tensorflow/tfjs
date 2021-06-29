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
            doc = "Common root path to remove when copying. Relative to the build file's directory",
        ),
        "srcs": attr.label_list(
            doc = "Files to copy",
        ),
    },
)
