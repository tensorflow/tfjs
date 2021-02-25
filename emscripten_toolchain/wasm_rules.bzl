"""Rules related to C++ and WebAssembly.
"""

load("//emscripten_toolchain:wasm_cc_binary.bzl", _wasm_cc_binary = "wasm_cc_binary")

wasm_cc_binary = _wasm_cc_binary
