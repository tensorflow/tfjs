load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _enumerate_tests_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.attr.name + ".ts")
    input_paths = [src.path for src in ctx.files.srcs]

    input_paths_file = ctx.actions.declare_file("enumerate_tests_paths")
    ctx.actions.write(input_paths_file, "\n".join(input_paths))

    run_node(
        ctx,
        executable = "enumerate_tests_bin",
        inputs = ctx.files.srcs + [input_paths_file],
        outputs = [output_file],
        arguments = [
            "-r",
            ctx.attr.root_path,
            "-o",
            output_file.path,
            input_paths_file.path,
        ],
    )

    return [DefaultInfo(files = depset([output_file]))]

enumerate_tests = rule(
    implementation = _enumerate_tests_impl,
    attrs = {
        "enumerate_tests_bin": attr.label(
            executable = True,
            cfg = "exec",
            default = Label("@//tools:enumerate_tests_bin"),
            doc = "The script that enumerates the tests",
        ),
        "root_path": attr.string(
            default = "",
            doc = "A root path to remove from srcs when importing them",
        ),
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = [".ts"],
            doc = "Test files to enumerate (i.e. import) in the output file",
        ),
    },
    doc = """Generates a ts file that imports the given test srcs

    This rule creates a test entrypoint that imports all of the individual tests
    specified in 'srcs'. In tfjs-core, it is used to create 'tests.ts', which
    other packages use to import the core tests all at once.
    """,
)
