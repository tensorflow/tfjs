load("@bazel_skylib//lib:paths.bzl", "paths")
load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _gen_json_impl(ctx):
    input_paths = [src.path for src in ctx.files.srcs]
    outputs = []
    for f in ctx.files.srcs:
        dest_path = paths.join(ctx.attr.output_path, f.basename + ".json")
        out = ctx.actions.declare_file(dest_path)
        outputs.append(out)

    run_node(
        ctx,
        executable = "gen_json_bin",
        inputs = ctx.files.srcs,
        outputs = outputs,
        arguments = [
            "--dest",
            ctx.attr.output_path,
        ],
    )

    return [DefaultInfo(files = depset(outputs))]

gen_json = rule(
    implementation = _gen_json_impl,
    attrs = {
        "deps": attr.label_list(),
        "gen_json_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tools:gen_json_bin"),
            doc = "The script that map typescript files to json",
        ),
        "output_path": attr.string(
            default = "",
            doc = "Output directory for the json files",
        ),
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = [".ts"],
            doc = "Typescript files to convert to json",
        ),
    },
    doc = """Convert the typescript files to json. This is used by converter to
    keep javascript script in sync with python converter in term of kernel
    definitions.
    """,
)
