"""wasm_cc_binary rule for compiling C++ targets to WebAssembly.
"""

def _wasm_transition_impl(settings, attr):
    _ignore = (settings, attr)

    features = list(settings["//command_line_option:features"])
    linkopts = list(settings["//command_line_option:linkopt"])

    if attr.threads == "emscripten":
        # threads enabled
        features.append("use_pthreads")
    elif attr.threads == "off":
        # threads disabled
        features.append("-use_pthreads")

    if attr.exit_runtime == True:
        features.append("exit_runtime")

    if attr.backend == "llvm":
        features.append("llvm_backend")
    elif attr.backend == "emscripten":
        features.append("-llvm_backend")

    if attr.simd:
        features.append("wasm_simd")

    return {
        "//command_line_option:compiler": "emscripten",
        "//command_line_option:crosstool_top": "//emscripten_toolchain:everything",
        "//command_line_option:cpu": "wasm",
        "//command_line_option:features": features,
        "//command_line_option:dynamic_mode": "off",
        "//command_line_option:linkopt": linkopts,
        "//command_line_option:platforms": [],
        "//command_line_option:custom_malloc": "//emscripten_toolchain:malloc",
    }

_wasm_transition = transition(
    implementation = _wasm_transition_impl,
    inputs = [
        "//command_line_option:features",
        "//command_line_option:linkopt",
    ],
    outputs = [
        "//command_line_option:compiler",
        "//command_line_option:cpu",
        "//command_line_option:crosstool_top",
        "//command_line_option:features",
        "//command_line_option:dynamic_mode",
        "//command_line_option:linkopt",
        "//command_line_option:platforms",
        "//command_line_option:custom_malloc",
    ],
)

def _wasm_binary_impl(ctx):
    cc_target = ctx.attr.cc_target[0]

    args = [
        "--output_path={}".format(ctx.outputs.loader.dirname),
    ] + [
        ctx.expand_location("--archive=$(location {})".format(
            cc_target.label,
        ), [cc_target]),
    ]
    outputs = [
        ctx.outputs.loader,
        ctx.outputs.wasm,
        ctx.outputs.map,
        ctx.outputs.mem,
        ctx.outputs.fetch,
        ctx.outputs.worker,
        ctx.outputs.data,
        ctx.outputs.symbols,
        ctx.outputs.dwarf,
    ]

    ctx.actions.run(
        inputs = ctx.files.cc_target,
        outputs = outputs,
        arguments = args,
        executable = ctx.executable._wasm_binary_extractor,
    )

    return DefaultInfo(
        files = depset(outputs),
        # This is needed since rules like web_test usually have a data
        # dependency on this target.
        data_runfiles = ctx.runfiles(transitive_files = depset(outputs)),
    )

def _wasm_binary_outputs(name, cc_target):
    basename = cc_target.name
    basename = basename.split(".")[0]
    outputs = {
        "loader": "{}/{}.js".format(name, basename),
        "wasm": "{}/{}.wasm".format(name, basename),
        "map": "{}/{}.wasm.map".format(name, basename),
        "mem": "{}/{}.js.mem".format(name, basename),
        "fetch": "{}/{}.fetch.js".format(name, basename),
        "worker": "{}/{}.worker.js".format(name, basename),
        "data": "{}/{}.data".format(name, basename),
        "symbols": "{}/{}.js.symbols".format(name, basename),
        "dwarf": "{}/{}.wasm.debug.wasm".format(name, basename),
    }

    return outputs

# Wraps a C++ Blaze target, extracting the appropriate files.
#
# This rule will transition to the emscripten toolchain in order
# to build the the cc_target as a WebAssembly binary.
#
# Args:
#   name: The name of the rule.
#   cc_target: The cc_binary or cc_library to extract files from.
wasm_cc_binary = rule(
    implementation = _wasm_binary_impl,
    attrs = {
        "backend": attr.string(
            default = "_default",
            values = ["_default", "emscripten", "llvm"],
        ),
        "cc_target": attr.label(
            cfg = _wasm_transition,
            mandatory = True,
        ),
        "exit_runtime": attr.bool(
            default = False,
        ),
        "threads": attr.string(
            default = "_default",
            values = ["_default", "emscripten", "off"],
        ),
        "simd": attr.bool(
            default = False,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
        "_wasm_binary_extractor": attr.label(
            executable = True,
            allow_files = True,
            cfg = "exec",
            default = Label("//emscripten_toolchain:wasm_binary"),
        ),
    },
    outputs = _wasm_binary_outputs,
)
