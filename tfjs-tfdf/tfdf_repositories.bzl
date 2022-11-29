# Copyright 2022 Google LLC.
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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tfdf_repositories(version = "1.1.0"):
    versions = {
        "1.1.0": "36b6974996d899589ba99ee95fb56699bf34c582f71e2d98475d7db4bce43b5b",
    }

    if not version in versions:
        versions_string = ", ".join(versions.keys())
        fail("Unsupported tfdf wasm files version %s. Supported versions are %s." %
             (version, versions_string))

    http_archive(
        name = "tfdf_wasm_files",
        sha256 = versions[version],
        url = "https://github.com/google/yggdrasil-decision-forests/releases/download/%s/javascript_wasm.zip" % version,
        build_file_content = """
filegroup(
    name = "wasm_files",
    visibility = ["//visibility:public"],
    srcs = [
        "inference.js",
        "inference.wasm",
    ],
)
""",
    )
