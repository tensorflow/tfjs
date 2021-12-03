load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tflite_repositories(version = "0.0.6"):
    http_archive(
        name = "tflite_wasm_files",
        sha256 = "0a5fbba206016265d9422b0c9d0514591b01306e392c4f9371cf1d4352d1161d",
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
