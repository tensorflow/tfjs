load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _gen_op_impl(ctx):
    #output_file_name = ctx.label.name + ".ts"
    #output_file = ctx.actions.declare_file(output_file_name)
    output_file = ctx.outputs.out

    run_node(
        ctx,
        executable = "gen_op_bin",
        inputs = [ctx.file.src],
        outputs = [output_file],
        arguments = [
            ctx.file.src.path,
            output_file.path,
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

gen_op = rule(
    implementation = _gen_op_impl,
    attrs = {
        "gen_op_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tfjs-converter/scripts:gen_op_bin"),
            doc = "The script that generates ts ops from json",
        ),
        "src": attr.label(
            allow_single_file = [".json"],
        ),
        "out": attr.output(
            mandatory = True,
            doc = "Output label for the generated .ts file",
        ),
    },
    #outputs = {"": "%{name}.js"},
)

