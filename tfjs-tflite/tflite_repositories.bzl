load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tflite_repositories(version = "0.0.6"):
    versions = {
        "0.0.3": "918dc8c5008cc57907315f78572b233abafefaf668c34e20596cedf96decf381",
        "0.0.4": "0d6f487cf7a5afd576417381130a0d9234f1d223012d75fe58eafe98177737b2",
        "0.0.5": "0a5fbba206016265d9422b0c9d0514591b01306e392c4f9371cf1d4352d1161d",
        "0.0.6": "a5f197e8c8c03bbf93659c9f2d48ddb8226a6a67795677dbdb82e2a9fde0b47c",
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
        "tflite_web_api_cc_simd.js",
        "tflite_web_api_cc_simd_threaded.js",
        "tflite_web_api_cc_simd_threaded.wasm",
        "tflite_web_api_cc_simd_threaded.worker.js",
        "tflite_web_api_cc_simd.wasm",
        "tflite_web_api_cc_threaded.js",
        "tflite_web_api_cc_threaded.wasm",
        "tflite_web_api_cc_threaded.worker.js",
        "tflite_web_api_cc.wasm",
        "tflite_web_api_client.js",
    ],
)
""",
    )
