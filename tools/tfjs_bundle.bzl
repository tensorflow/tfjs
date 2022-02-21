# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@npm//@bazel/rollup:index.bzl", "rollup_bundle")

def _make_rollup_config_impl(ctx):
    output_file = ctx.actions.declare_file(ctx.label.name + ".js")

    # TODO(mattsoulanille): This stats file is not declared as an output
    # of any rule. It should be declared as an output of the rollup_bundle
    # rule, but there doesn't seem to be an easy way to do that.
    stats_file_path = paths.join(
        ctx.bin_dir.path,
        paths.dirname(ctx.build_file_path),
        ctx.attr.stats,
    )

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.config_file,
        substitutions = {
            "TEMPLATE_es5": "true" if ctx.attr.es5 else "false",
            "TEMPLATE_minify": "true" if ctx.attr.minify else "false",
            "TEMPLATE_stats": stats_file_path,
        },
    )
    return [DefaultInfo(files = depset([output_file]))]

_make_rollup_config = rule(
    implementation = _make_rollup_config_impl,
    attrs = {
        "es5": attr.bool(
            default = False,
            doc = "Whether to transpile to es5",
            mandatory = False,
        ),
        "minify": attr.bool(
            default = False,
            doc = "Whether to minify with terser",
            mandatory = False,
        ),
        "stats": attr.string(
            mandatory = True,
            doc = "The name of the stats file",
        ),
        "template": attr.label(
            default = Label("@//tools:rollup_template.config.js"),
            allow_single_file = True,
            doc = "The karma config template to expand",
        ),
    },
    outputs = {"config_file": "%{name}.js"},
)

def tfjs_rollup_bundle(name, deps, entry_point, minify = False, umd_name = None, es5 = False, **kwargs):
    config_file = name + "_config"
    _make_rollup_config(
        name = config_file,
        stats = name + "_stats.html",
        es5 = es5,
        minify = minify,
    )

    rollup_deps = deps + [
        "@npm//@rollup/plugin-commonjs",
        "@npm//@rollup/plugin-node-resolve",
        "@npm//rollup-plugin-sourcemaps",
        "@npm//rollup-plugin-terser",
        "@npm//rollup-plugin-visualizer",
        "@npm//typescript",
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
        **kwargs
    )

def tfjs_bundle(name, deps, entry_point, umd_name, external = [], testonly = False, **kwargs):
    # A note on minification: While it would be more efficient to create
    # unminified bundles and then run them through terser separately, that
    # would prevent us from creating bundle visualizations for minified bundles
    # with rollup-plugin-visualizer. If / when there is a standalone visualizer
    # available, it may make sense to switch to it, since that would make this
    # build more efficient and would simplify some of the rules used.
    for minify in [True, False]:
        dot_min = ".min" if minify else ""

        # UMD ES2017
        tfjs_rollup_bundle(
            name = name + ".es2017" + dot_min,
            testonly = testonly,
            deps = deps,
            entry_point = entry_point,
            umd_name = umd_name,
            format = "umd",
            minify = minify,
        )

        # UMD es5
        tfjs_rollup_bundle(
            name = name + dot_min,
            testonly = testonly,
            deps = deps,
            entry_point = entry_point,
            umd_name = umd_name,
            format = "umd",
            minify = minify,
            es5 = True,
        )

        # FESM ES2017
        # TODO(mattsoulanille): Check that this is actually
        # generating flat esm modules.
        tfjs_rollup_bundle(
            name = name + ".fesm" + dot_min,
            testonly = testonly,
            deps = deps,
            entry_point = entry_point,
            format = "esm",
            minify = minify,
        )

        # cjs es5 node bundle
        tfjs_rollup_bundle(
            name = name + ".node" + dot_min,
            testonly = testonly,
            deps = deps,
            entry_point = entry_point,
            format = "cjs",
            minify = minify,
            es5 = True,
        )
