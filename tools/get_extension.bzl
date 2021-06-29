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

def _get_extension_impl(ctx):
    outfiles = [f for s in ctx.attr.srcs for f in s.files.to_list() if f.extension == ctx.attr.extension]

    return [DefaultInfo(files = depset(outfiles))]

get_extension = rule(
    implementation = _get_extension_impl,
    attrs = {
        "extension": attr.string(
            doc = "File extension to get",
        ),
        "srcs": attr.label_list(
            doc = "Sources to get extensions from",
        ),
    },
)
