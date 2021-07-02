load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _enumerate_tests_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".ts")
    input_paths = [src.path for src in ctx.files.srcs]

    run_node(
        ctx,
        executable = "enumerate_tests_bin",
        inputs = ctx.files.srcs,
        outputs = [output_file],
        arguments = [
            '-r', ctx.attr.root_path,
            '-o', output_file.path,
        ] + input_paths,
    )

    return [DefaultInfo(files = depset([output_file]))]

enumerate_tests = rule(
    implementation = _enumerate_tests_impl,
    attrs = {
        "enumerate_tests_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tools:enumerate_tests_bin"),
            doc = "The script that enumerates the tests",
        ),
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = [".ts"],
            doc = "Test files to enumerate (i.e. import) in the output file",
        ),
        "root_path": attr.string(
            default = "",
            doc = "A root path to remove from srcs when importing them",
        ),
    },
    doc = """Generates a ts file that imports the given test srcs

    This rule creates a test entrypoint that imports all of the individual tests
    specified in 'srcs'. In tfjs-core, it is used to create 'tests.ts', which
    other packages use to import the core tests all at once.
    """
)
