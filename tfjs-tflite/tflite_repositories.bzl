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

def tflite_repositories(version = "0.0.9"):
    versions = {
        "0.0.3": "918dc8c5008cc57907315f78572b233abafefaf668c34e20596cedf96decf381",
        "0.0.4": "0d6f487cf7a5afd576417381130a0d9234f1d223012d75fe58eafe98177737b2",
        "0.0.5": "0a5fbba206016265d9422b0c9d0514591b01306e392c4f9371cf1d4352d1161d",
        "0.0.6": "a5f197e8c8c03bbf93659c9f2d48ddb8226a6a67795677dbdb82e2a9fde0b47c",
        "0.0.7": "ffedda0e96485adcb0fde272320e2cc121202fdbaa403dc39eaea8c1e6fac84e",
        "0.0.8": "cb3c4ee99aacaba325dedca837c5a83828aeeb2adbcae43179a24a3e819cbcc7",
        "0.0.9": "25819b6841dc460e43f4e43746eae3b89d6a372615cfff726495d3910fce9178",
    }

    if not version in versions:
        versions_string = ", ".join(versions.keys())
        fail("Unsupported tflite web api version %s. Supported versions are %s." %
             (version, versions_string))

    http_archive(
        name = "tflite_wasm_files",
        sha256 = versions[version],
        url = "https://storage.googleapis.com/tfweb/%s/tflite_web_api.zip" % version,
        build_file_content = """
filegroup(
    name = "wasm_files",
    visibility = ["//visibility:public"],
    srcs = [
        "tflite_web_api_cc.js",
        "tflite_web_api_cc.wasm",
        "tflite_web_api_cc_simd.js",
        "tflite_web_api_cc_simd_threaded.js",
        "tflite_web_api_cc_simd_threaded.wasm",
        "tflite_web_api_cc_simd_threaded.worker.js",
        "tflite_web_api_cc_simd.wasm",
        "tflite_web_api_cc_threaded.js",
        "tflite_web_api_cc_threaded.wasm",
        "tflite_web_api_cc_threaded.worker.js",
        "tflite_web_api_client.js",
    ],
)
""",
    )
