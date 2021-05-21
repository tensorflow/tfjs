load("@bazel_skylib//lib:paths.bzl", "paths")


def _copy_to_dist_impl(ctx):
    files = [f for s in ctx.attr.srcs for f in s.files.to_list()]

    root_dir = paths.join(paths.dirname(ctx.build_file_path), ctx.attr.root)

    outputs = []
    for f in files:
        dest_path = paths.join("dist", paths.relativize(f.short_path, root_dir))
        out = ctx.actions.declare_file(dest_path)
        outputs.append(out)
        ctx.actions.symlink(
            output = out,
            target_file = f,
        )
    return [DefaultInfo(
        files = depset(outputs)
    )]

copy_to_dist = rule(
    implementation = _copy_to_dist_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "Files to copy",
        ),
        "root": attr.string(
            default = "",
            doc = "Common root path to remove when copying. Relative to the build file's directory",
        ),            
    }
)

    
