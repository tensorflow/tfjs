# Copyright 2022 Google LLC. All Rights Reserved.
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

def _get_extension_impl(ctx):
    # Starlark does not support sets, so use a dict instead.
    included = {}
    if ctx.attr.include:
        for extension in ctx.attr.include:
            for f in ctx.files.srcs:
                if f.basename.endswith(extension):
                    included[f] = True
    else:
        included = {src: True for src in ctx.files.srcs}

    if ctx.attr.exclude:
        for extension in ctx.attr.exclude:
            for f in ctx.files.srcs:
                if f.basename.endswith(extension):
                    included.pop(f)

    return DefaultInfo(files = depset(included.keys()))

get_extension = rule(
    implementation = _get_extension_impl,
    attrs = {
        "exclude": attr.string_list(
            mandatory = False,
            doc = "File extensions to exclude.",
        ),
        "include": attr.string_list(
            mandatory = False,
            doc = "File extensions to include. If empty, includes all files.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
    },
    doc = """Select files with a given file extension

    Creates a target referencing a set of files that match the
    include and exclude attributes.
    """,
)
