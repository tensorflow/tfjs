load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _kernels_to_ops_impl(ctx):
    output_file = ctx.outputs.out

    run_node(
        ctx,
        executable = "kernels_to_ops_bin",
        inputs = ctx.files.srcs,
        outputs = [output_file],
        chdir = "tfjs-converter",
        arguments = [
            "--out",
            "../" + output_file.path, # '../' due to chdir above
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

kernels_to_ops = rule(
    implementation = _kernels_to_ops_impl,
    attrs = {
        "kernels_to_ops_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tfjs-converter/scripts:kernels_to_ops_bin"),
            doc = "The script that generates the kernel2op.json metadata file",
        ),
        "srcs": attr.label_list(
            doc = "The files in the ts project",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "Output label for the generated .json file",
        ),
    },
)
