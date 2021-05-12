load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
load("@npm//@bazel/terser:index.bzl", "terser_minified")
load("@npm//@bazel/typescript:index.bzl", "ts_project")


def tfjs_rollup_bundle(name, deps, entry_point, umd_name=None, es5=False, **kwargs):
    rollup_deps = deps + [
       "@npm//@rollup/plugin-commonjs",
       "@npm//@rollup/plugin-node-resolve",
       "@npm//rollup-plugin-sourcemaps",
    ]

    config_file = "@//bundling:rollup.config.js"
    srcs = kwargs.pop('srcs', [])
    if es5:
        rollup_deps += [
            "@npm//@rollup/plugin-babel",
        ]
        config_file = "@//bundling:rollup.es5.config.js"
        srcs += [
            "@//:babel.config.json",
        ]

    rollup_args = []
    if umd_name:
        rollup_args += ["--output.name={}".format(umd_name)]

    rollup_bundle(
        name = name,
        args = rollup_args,
        sourcemap = "true",
        entry_point = entry_point,
        config_file = config_file,
        deps = rollup_deps,
        srcs = srcs,
        **kwargs,
    )

def tfjs_bundle(name, deps, entry_point, umd_name, external = [], testonly = False, **kwargs):
    # UMD ES2017
    tfjs_rollup_bundle(
        name = name + ".es2017",
        testonly = testonly,
        deps = deps,
        entry_point = entry_point,
        umd_name = umd_name,
        format = "umd",
    )

    # UMD es5
    tfjs_rollup_bundle(
        name = name,
        testonly = testonly,
        deps = deps,
        entry_point = entry_point,
        umd_name = umd_name,
        format = "umd",
        es5 = True,
    )

    # FESM ES2017
    # TODO(mattsoulanille): Check that this is actually
    # generating flat esm modules.
    tfjs_rollup_bundle(
        name = name + ".fesm",
        testonly = testonly,
        deps = deps,
        entry_point = entry_point,
        format = "esm",
    )

    # cjs es5 node bundle
    tfjs_rollup_bundle(
        name = name + ".node",
        testonly = testonly,
        deps = deps,
        entry_point = entry_point,
        format = "cjs",
        es5 = True,
    )

    # Minified bundles
    for extension in ["", ".es2017", ".fesm", ".node"]:
        src = name + extension
        terser_minified(
            name = src + ".min",
            src = src,
        )
