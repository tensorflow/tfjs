load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _create_worker_module_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.out_file)
    if (len(ctx.files.worker_file) != 1):
        fail("Expected a single file but got " + str(ctx.files.worker_file))

    cjs = ["--cjs"] if ctx.attr.cjs else []
    run_node(
        ctx,
        executable = "create_worker_module_bin",
        inputs = ctx.files.worker_file,
        outputs = [output_file],
        arguments = cjs + [
            ctx.files.worker_file[0].path,
            output_file.path,
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

create_worker_module = rule(
    implementation = _create_worker_module_impl,
    attrs = {
        "cjs": attr.bool(
            default = False,
            doc = "Whether to output commonjs",
        ),
        "create_worker_module_bin": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@//tfjs-backend-wasm/scripts:create_worker_module_bin"),
            doc = "The script that creates the worker module",
        ),
        "out_file": attr.string(
            mandatory = True,
            doc = "The name for the output file",
        ),
        "worker_file": attr.label(
            doc = "The worker file to transform",
            allow_files = [".js"],
        ),
    },
    doc = """Modify the Emscripten WASM worker script so it can be inlined

    ...by the tf-backend-wasm bundle.
    """,
)
