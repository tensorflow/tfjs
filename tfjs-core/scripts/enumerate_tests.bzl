load("@build_bazel_rules_nodejs//:providers.bzl", "run_node")

def _enumerate_tests_impl(ctx):
    output_file = ctx.actions.declare_file("tests.ts")
    run_node(
        ctx,
        executable = "enumerate_tests_bin",
        inputs = ctx.files.srcs,
        outputs = [output_file],
        arguments = [],
        chdir = "tfjs-core",
    )

    return [DefaultInfo(files = depset([output_file]))]

enumerate_tests = rule(
    implementation = _enumerate_tests_impl,
    attrs = {
        "enumerate_tests_bin": attr.label(
            executable = True,
            cfg = "host",
            default = Label("@//tfjs-core/scripts:enumerate_tests_bin"),
        ),
        "srcs": attr.label_list(mandatory = True, allow_files = [".ts"]),
    },
)
