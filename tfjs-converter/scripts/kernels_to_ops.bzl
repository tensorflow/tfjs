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

load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _kernels_to_ops_impl(ctx):
    output_file = ctx.outputs.out

    run_node(
        ctx,
        executable = "kernels_to_ops_bin",
        inputs = ctx.files.srcs,
        outputs = [output_file],
        chdir = "tfjs-converter",
        arguments = [
            "--out",
            "../" + output_file.path,  # '../' due to chdir above
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

kernels_to_ops = rule(
    implementation = _kernels_to_ops_impl,
    attrs = {
        "kernels_to_ops_bin": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@//tfjs-converter/scripts:kernels_to_ops_bin"),
            doc = "The script that generates the kernel2op.json metadata file",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "Output label for the generated .json file",
        ),
        "srcs": attr.label_list(
            doc = "The files in the ts project",
        ),
    },
)
