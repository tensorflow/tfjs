load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _patch_threaded_simd_module_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.out_file)
    if (len(ctx.files.threaded_simd_file) != 1):
        fail("Expected a single file but got " + str(ctx.files.threaded_simd_file))

    run_node(
        ctx,
        executable = "patch_threaded_simd_module_bin",
        inputs = ctx.files.threaded_simd_file,
        outputs = [output_file],
        arguments = [
            ctx.files.threaded_simd_file[0].path,
            output_file.path,
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

patch_threaded_simd_module = rule(
    implementation = _patch_threaded_simd_module_impl,
    attrs = {
        "out_file": attr.string(
            mandatory = True,
            doc = "The name for the output file",
        ),
        "patch_threaded_simd_module_bin": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@//tfjs-backend-wasm/scripts:patch_threaded_simd_module_bin"),
            doc = "The script that creates the worker module",
        ),
        "threaded_simd_file": attr.label(
            doc = "The threaded simd file to transform",
            allow_files = [".js"],
        ),
    },
    doc = """Fix the check for _scriptDir. See patch-threaded-simd-module.js
    
    ...for more details.
    """,
)
