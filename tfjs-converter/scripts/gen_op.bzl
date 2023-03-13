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
load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _gen_op_impl(ctx):
    files = [f for s in ctx.attr.srcs for f in s.files.to_list()]
    outputs = []

    for f in files:
        ts_filename = f.basename[:-len(f.extension)] + "ts"
        dest_path = paths.join(ctx.attr.dest_dir, ts_filename)
        output_file = ctx.actions.declare_file(dest_path)
        outputs.append(output_file)
        run_node(
            ctx,
            executable = "gen_op_bin",
            inputs = [f],
            outputs = [output_file],
            arguments = [
                f.path,
                output_file.path,
            ],
        )

    return [DefaultInfo(files = depset(outputs))]

gen_op = rule(
    implementation = _gen_op_impl,
    attrs = {
        "dest_dir": attr.string(
            mandatory = True,
            doc = "Output directory for the generated .ts files relative to the BUILD file",
        ),
        "gen_op_bin": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@//tfjs-converter/scripts:gen_op_bin"),
            doc = "The script that generates ts ops from json",
        ),
        "srcs": attr.label_list(
            allow_files = [".json"],
            doc = "The json files to generate the ops from",
        ),
    },
)
