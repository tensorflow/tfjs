load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _make_version_test_file_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".ts")

    run_node(
        ctx,
        executable = "make_version_test_file_bin",
        inputs = ctx.files.package_json,
        outputs = [output_file],
        arguments = [
            ctx.files.package_json[0].path,
            output_file.path,
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

make_version_test_file = rule(
    implementation = _make_version_test_file_impl,
    attrs = {
        "make_version_test_file_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tools:make_version_test_file_bin"),
            doc = "The script that generates the version test",
        ),
        "package_json": attr.label(
            mandatory = True,
            allow_single_file = [".json"],
            doc = "The package.json containing the version to use for the test",
        ),
    },
    doc = """Generate the version test file for tfjs-core""",
)
