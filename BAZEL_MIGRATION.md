# Bazel Migration

This document details the steps to migrate a package to build with Bazel. These steps are easiest to understand with a working example, so this doc references `tfjs-core`'s setup as much as possible. Since this migration is still in progress, the steps and processes listed here may change as we improve on the process, add features to each package's build, and create tfjs-specific build functions.

## Scope
Migrating a package to Bazel involves adding Bazel targets that build the package, running its tests, and packing the package for publishing to npm. To ease the transition to Bazel, we're incrementally transitioning packages to build with Bazel, starting with root packages (tfjs-core, tfjs-backend-cpu) and gradually expanding to leaf packages. This is different from our original approach of maintaining our current build and a new Bazel build in parallel, which ended up not working due to some changes that Bazel required to the ts sources.

## Caveats
- Bazel will only make a dependency or file available to the build if you explicitly declare it as a dependency / input to the rule you're using.
- All Bazel builds use the root package.json, so you may have to add packages to it. As long as you use `@npm//dependency-name` in BUILD files to add dependencies, you won't need to worry about the build accidentally seeing the package's `node_modules` directory instead of the root `node_modules`. Bazel will only make the root `node_modules` directory visible to the build.
  - Even though the build doesn't use the package's `node_modules`, you may have to run `yarn` within the package to get code completion to work correctly. We're looking into why this is the case.
- There may be issues with depending on explicitly pinned versions of `@tensorflow` scoped packages, which might affect some demos if they're migrated to use Bazel. We might just want to leave demos out of Bazel so they're easier to understand.

## Steps
These steps are general guidelines for how to build a package with Bazel. They should work for most packages, but there may be some exceptions (e.g. wasm, react native).

### Make sure all dependencies build with Bazel
A package's dependencies must be migrated before it can be migrated.

### Create a `BUILD.bazel` file in the package
Bazel looks for targets to run in `BUILD` and `BUILD.bazel` files. Use the `.bazel` extension since blaze uses `BUILD`. You may want to install an extension for your editor to get syntax highlighting. [Here's the vscode extension](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel).

### Compile the package with `ts_library`
[ts_library](https://bazelbuild.github.io/rules_nodejs/TypeScript.html#ts_library) compiles the package. We use a custom version of `ts_library` that sets some project-specific settings.

Here's an example of how `tfjs-core` uses `ts_library` to build.

`tfjs-core/src/BUILD.bazel`
```starlark
load("//tools:defaults.bzl", "ts_library")

TEST_SRCS = [
    "**/*_test.ts",
    "image_test_util.ts",
]

# Compiles the majority of tfjs-core using the `@tensorflow/tfjs-core/dist`
# module name.
ts_library(
    name = "tfjs-core_src_lib",
    srcs = glob(
        ["**/*.ts"],
        exclude = TEST_SRCS + ["index.ts"],
    ),
    module_name = "@tensorflow/tfjs-core/dist",
    deps = [
        "@npm//@types",
        "@npm//jasmine-core",
        "@npm//seedrandom",
    ],
)

# Compiles the `index.ts` entrypoint of tfjs-core separately from the rest of
# the sources in order to use the `@tensorflow/tfjs-core` module name instead
# of `@tensorflow/tfjs-core/dist`,
ts_library(
    name = "tfjs-core_lib",
    srcs = ["index.ts"],
    module_name = "@tensorflow/tfjs-core",
    deps = [
        ":tfjs-core_src_lib",
    ],
)
```
`ts_library` is used twice in order to have the correct `module_name` for the output files. Most files are imported relative to `@tensorflow/tfjs-core/src/`, but `index.ts`, the entrypoint of `tfjs-core`, should be importable as `@tensorflow/tfjs-core`.

### Bundle the package
This step involves bundling the compiled files from the compilation step into a single file. TFJS generates several bundles for each package in order to support different execution environments. We provide a `tfjs_bundle` macro to generate these bundles.

`tfjs-core/BUILD.bazel`
```starlark
load("//tools:tfjs_bundle.bzl", "tfjs_bundle")

tfjs_bundle(
    name = "tf-core",
    entry_point = "//tfjs-core/src:index.ts",
    external = [
        "node-fetch",
        "util",
    ],
    umd_name = "tf",
    deps = [
        "//tfjs-core/src:tfjs-core_lib",
        "//tfjs-core/src:tfjs-core_src_lib",
    ],
)
```
The `tfjs_bundle` macro generates several different bundles which are published in the package publishing step.


### Compile the tests with `ts_library`
We compile the tests with `ts_library`. In the case of `tfjs-core`, we actually publish the test files, since other packages use them in their tests. Therefore, it's important that we set the `module_name` to `@tensorflow/tfjs-core/dist`. In a future major version of tfjs, we may stop publishing the tests to npm.

`tfjs-core/src/BUILD.bazel`
```starlark
load("//tools:defaults.bzl", "ts_library")

ts_library(
    name = "tfjs-core_test_lib",
    srcs = glob(TEST_SRCS),
    # TODO(msoulanille): Mark this as testonly once it's no longer needed in the
    # npm package (for other downstream packages' tests).
    module_name = "@tensorflow/tfjs-core/dist",
    deps = [
        ":tfjs-core_lib",
        ":tfjs-core_src_lib",
    ],
)
```

### Run node tests with nodejs_test
Our test setup allows fine-tuning of exactly what tests are run via `setTestEnvs` and `setupTestFilters` in `jasmine_util.ts`, which are used in a custom Jasmine entrypoint file `setup_test.ts`. This setup does not work well with [jasmine_node_test](https://bazelbuild.github.io/rules_nodejs/Jasmine.html#jasmine_node_test), which provides its own entrypoint for starting Jasmine. Instead, we use the [nodejs_test](https://bazelbuild.github.io/rules_nodejs/Built-ins.html#nodejs_test) rule.

`tfjs-core/BUILD.bazel`
```starlark
load("@build_bazel_rules_nodejs//:index.bzl", "js_library", "nodejs_test")

# This is necessary for tests to have acess to
# the package.json so src/version_test.ts can 'require()' it.
js_library(
    name = "package_json",
    srcs = [
        ":package.json",
    ],
)

nodejs_test(
    name = "tfjs-core_node_test",
    data = [
        ":package_json",
        "//tfjs-backend-cpu/src:tfjs-backend-cpu_lib",
        "//tfjs-core/src:tfjs-core_lib",
        "//tfjs-core/src:tfjs-core_src_lib",
        "//tfjs-core/src:tfjs-core_test_lib",
    ],
    entry_point = "//tfjs-core/src:test_node.ts",
    link_workspace_root = True,
    tags = ["ci"],
)
```
It's important to tag tests with `ci` if you would like them to run in continuous integration.


### Run browser tests with karma_web_test
We use `esbuild` to bundle the tests into a single file.

`tfjs-core/src/BUILD.bazel`
```starlark
load("//tools:defaults.bzl", "esbuild")

esbuild(
    name = "tfjs-core_test_bundle",
    testonly = True,
    entry_point = "setup_test.ts",
    external = [
        # webworker tests call 'require('@tensorflow/tfjs')', which
        # is external to the test bundle.
        # Note: This is not a bazel target. It's just a string.
        "@tensorflow/tfjs",
        "worker_threads",
        "util",
    ],
    sources_content = True,
    deps = [
        ":tfjs-core_lib",
        ":tfjs-core_test_lib",
        "//tfjs-backend-cpu/src:tfjs-backend-cpu_lib",
        "//tfjs-core:package_json",
    ],
)
```

`tfjs-core/BUILD.bazel`

The esbuild bundle is then used in the tfjs_web_test macro, which uses [karma_web_test](https://bazelbuild.github.io/rules_nodejs/Concatjs.html#karma_web_test) to serve them to a browser to be run. Different browserstack browsers can be enabled or disabled in the `browsers` argument, and the full list of browsers is located in `tools/karma_template.conf.js`. Browserstack browser tests are automatically tagged with `ci`.

```starlark
load("//tools:tfjs_web_test.bzl", "tfjs_web_test")

tfjs_web_test(
    name = "tfjs-core_test",
    srcs = [
        "//tfjs-core/src:tfjs-core_test_bundle",
    ],
    browsers = [
        "bs_chrome_mac",
        "bs_firefox_mac",
        "bs_safari_mac",
        "bs_ios_11",
        "bs_android_9",
        "win_10_chrome",
    ],
    static_files = [
        # Listed here so sourcemaps are served
        "//tfjs-core/src:tfjs-core_test_bundle",
        # For the webworker
        ":tf-core.min.js",
        ":tf-core.min.js.map",
        "//tfjs-backend-cpu:tf-backend-cpu.min.js",
        "//tfjs-backend-cpu:tf-backend-cpu.min.js.map",
    ],
)
```

### Package for npm
We use the [pkg_npm](https://bazelbuild.github.io/rules_nodejs/Built-ins.html#pkg_npm) rule to create and publish the package to npm. However, there are a few steps needed before we can declare the package. For most packages, we distribute all our compiled outputs in the `dist` directory. However, due to how `ts_library` works, it creates outputs in the same directory as the source files were compiled from (except they show up in Bazel's `dist/bin` output dir). We need to copy these from `src` to `dist` while making sure Bazel is aware of this copy (so we can still use `pkg_npm`).

We also need to copy several other files to `dist`, such as the bundles created by `tfjs_bundle`, and we need to create [miniprogram](https://walkthechat.com/wechat-mini-programs-simple-introduction/) files for WeChat.

To copy files, we use the `copy_to_dist` rule. This rule creates symlinks to all the files in `srcs` and places them in a filetree with the same structure in `dest_dir` (which defaults to `dist`).

However, we can't just copy the output of a `ts_library`, since its default output is the `.d.ts` declaration files. We need to extract the desired ES Module and CommonJS outputs of the rule by selecting the appropriate [output groups of ts_library](https://bazelbuild.github.io/rules_nodejs/TypeScript.html#accessing-javascript-outputs) with `filegroup` rules. The output groups are named `es6_sources` and `es5_sources`, but those names don't actually mean they're es6 and es5 javascript and are just an artifact of how `ts_library` was first written. Both outputs are actually es2017, as configured in `tools/defaults.bzl` and differ only in what module system they use.

```starlark
# ES Module es2017 compilation results. This is configured in tools/defaults.bzl
# Outputs '.mjs' files
filegroup(
    name = "es6_sources",
    srcs = [
        "//tfjs-core/src:tfjs-core_lib",
        "//tfjs-core/src:tfjs-core_src_lib",
        "//tfjs-core/src:tfjs-core_test_lib",
    ],
    output_group = "es6_sources",
)

# Commonjs es2017 compilation results. Outputs '.js' files
filegroup(
    name = "es5_sources",
    srcs = [
        "//tfjs-core/src:tfjs-core_lib",
        "//tfjs-core/src:tfjs-core_src_lib",
        "//tfjs-core/src:tfjs-core_test_lib",
    ],
    output_group = "es5_sources",
)
```

Once we have filegroups pointing to the output `.js` and `.mjs` files, we can use `copy_to_dist` to copy them.

```starlark
load("//tools:copy_to_dist.bzl", "copy_to_dist")

copy_to_dist(
    name = "copy_src_to_dist",
    srcs = [
        ":es5_sources",
        ":es6_sources",
        "//tfjs-core/src:tfjs-core_lib", # Copy .d.ts files as well
        "//tfjs-core/src:tfjs-core_src_lib",
        "//tfjs-core/src:tfjs-core_test_lib",
    ],
    root = "src",
)
```

We can also copy the bundles output from `tfjs_bundle`

```starlark
copy_to_dist(
    name = "copy_bundles",
    srcs = [
        ":tf-core",
        ":tf-core.es2017",
        ":tf-core.es2017.min",
        ":tf-core.fesm",
        ":tf-core.fesm.min",
        ":tf-core.min",
    ],
)
```

We copy the miniprogram files as well, this time using the `copy_file` rule, which copies a single file to a destination.

```starlark
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

copy_file(
    name = "copy_miniprogram",
    src = ":tf-core.min.js",
    out = "dist/miniprogram/index.js",
)

copy_file(
    name = "copy_miniprogram_map",
    src = ":tf-core.min.js.map",
    out = "dist/miniprogram/index.js.map",
)
```

Now that all the files are copied, we can declare a `pkg_npm`

```starlark
load("@build_bazel_rules_nodejs//:index.bzl", "pkg_npm")

pkg_npm(
    name = "tfjs-core_pkg",
    srcs = ["package.json"],
    tags = ["ci"],
    deps = [
        ":copy_bundles",
        ":copy_miniprogram",
        ":copy_miniprogram_map",
        ":copy_src_to_dist",
        ":copy_test_snippets", # <- This is only in core
    ],
)
```

Now the package can be published to npm with `bazel run //tfjs-core:tfjs-core_pkg.publish`.


### Publishing to npm
With a `pkg_npm` rule defined, we add a script to `package.json` to run it. This script will be used by the main script that publishes the monorepo. 

```json
"scripts" {
    "publish-npm": "bazel run :tfjs-core_pkg.publish"
}
```

Since we now use the `publish-npm` script to publish this package instead of `npm publish`, we need to make sure the release tests and release script know how to publish it.

1. In `scripts/publish-npm.ts`, add your package's name to the `BAZEL_PACKAGES` set.
2. In `e2e/scripts/publish-tfjs-ci.sh`, add your package's name to the `BAZEL_PACKAGES` list.
