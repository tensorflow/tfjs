def _get_extension_impl(ctx):
    outfiles = [f for s in ctx.attr.srcs for f in s.files.to_list() if f.extension == ctx.attr.extension]

    return [DefaultInfo(files = depset(outfiles))]

get_extension = rule(
    implementation = _get_extension_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "Sources to get extensions from",
        ),
        "extension": attr.string(
            doc = "File extension to get"
        ),
    }
)
