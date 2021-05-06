load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")
#load("@npm//@bazel/terser:index.bzl", "terser_minified")
load("//:esbuild.bzl", "esbuild")
load("@npm//typescript:index.bzl", "tsc")
load("@npm//@babel/cli:index.bzl", "babel")
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

def es5(name, testonly, src):
    tsc(
        name = name,
        testonly = testonly,
        outs = [
            name + ".js",
            name + ".js.map",
        ],
        args = [
            "$(execpath :{})".format(src),
            "--types",
            "--skipLibCheck",
            "--target",
            "es5",
            "--lib",
            "es2015,dom",
            "--allowJS",
            "--sourcemap",
            "--outFile",
            "$(execpath :{}.js)".format(name),
        ],
        data = [
            src,
            src + ".map",
        ],
    )

def es5_babel(name, testonly, src):
    babel(
        name = name,
        outs = [
            name + ".js",
            name + ".js.map",
        ],
        args = [
            "$(execpath :{})".format(src),
            "--config-file",
            "./$(execpath @//bundling:es5.babelrc)",
            "--source-maps",
            "true",
            "--out-file",
            "$(execpath :{}.js)".format(name),
        ],
        data = [
            src,
            src + ".map",
            "@//bundling:es5.babelrc",
            "@npm//@babel/preset-env",
        ],
    )


def tfjs_ts_project(name, srcs, **kwargs):
    ts_project(
        name = name,
        srcs = srcs,
        declaration = True,
        extends = "@//:tsconfig.json",
        incremental = True,
        out_dir = "dist",
        source_map = True,
        tsconfig = {
            "compilerOptions": {
                "target": "es2017",
            },
        },
        **kwargs,
    )

    # ts_project(
    #     name = name + "_es5",
    #     srcs = srcs,
    #     declaration = True,
    #     extends = "@//:tsconfig.json",
    #     incremental = True,
    #     out_dir = "dist_es5",
    #     source_map = True,
    #     tsconfig = {
    #         "compilerOptions": {
    #             "target": "es5",
    #         },
    #     },
    #     **kwargs
    # )


def tfjs_bundle(name, deps, entry_point, umd_name, external = [], testonly = False, **kwargs):
    # # es5 sources
    # native.filegroup(
    #     name = name + "_es5",
    #     srcs = deps,
    #     # Change to es6_sources to get the 'prodmode' JS
    #     output_group = "es5_sources",
    # )

    # # es2017 sources
    # native.filegroup(
    #     name = name + "_es2017",
    #     srcs = deps,
    #     output_group = "es6_sources",
    # )

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

    # # es5 transformation applied to UMD bundles
    # es5_babel(
    #     name = name,
    #     testonly = testonly,
    #     src = name + ".es2017.js",
    # )

    # FESM ES2017
    # TODO(mattsoulanille): Check that this is actually
    # generating flat esm modules and using es2017.
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
        #deps = [name + "_es5"],
        deps = deps,
        #entry_point = es5_entry_point,
        entry_point = entry_point,
        format = "cjs",
        es5 = True,
    )

    # cjs bundle
    # tfjs_rollup_bundle(
    #     name = name + ".node",
    #     testonly = testonly,
    #     deps = deps,
    #     entry_point = entry_point,
    #     format = "cjs",
    #     es5 = True,
    # )

    # # es5 node bundle
    # es5_babel(
    #     name = name + ".node",
    #     testonly = testonly,
    #     src = name + ".cjs.js",
    # )

    # Minified bundles
    for extension in ["", ".es2017", ".fesm", ".cjs", ".node"]:
        src = name + extension
        terser_minified(
            name = src + ".min",
            src = src,
        )
            
