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
A package's dependencies must be migrated before it can be migrated. Take a look at the package's issue, which can be found by checking [#5287](https://github.com/tensorflow/tfjs/issues/5287), to find its dependencies.

### Add dependencies to the root `package.json`
Bazel (through `rules_nodejs`) uses a single root `package.json` for its npm dependencies. When converting a package to build with Bazel, dependencies in the package's `package.json` will need to be added to the root `package.json` as well.

### Create a `BUILD.bazel` file in the package's root
Bazel looks for targets to run in `BUILD` and `BUILD.bazel` files. Use the `.bazel` extension since blaze uses `BUILD`. You may want to install an extension for your editor to get syntax highlighting. [Here's the vscode extension](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel).

This BUILD file will handle package-wide rules like bundling for npm.

### Create another `BUILD.bazel` file in `src`
This BUILD file will compile the source files of the package using `ts_library` and may also define test bundles.

### Compile the package with `ts_library`
In the `src` BUILD.bazel file, we use `ts_library` to compile the package's typescript files.  [ts_library](https://bazelbuild.github.io/rules_nodejs/TypeScript.html#ts_library) is a rule provided by rules_nodejs. We wrap `ts_library` in a macro that sets some project-specific settings.

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

If your package imports from `dist` (e.g. `import {} from @tensorflow/tfjs-core/dist/ops/ops_for_converter`), that import likely corresponds to a rule in that packages `src/BUILD.bazel` file. Look for a rule that includes the file you're importing and has `module_name` set correctly for that import.

### Bundle the package
This step involves bundling the compiled files from the compilation step into a single file, and the rules are added to the package's root BUILD file (instead of `src/BUILD.bazel`). In order to support different execution environments, TFJS generates several bundles for each package. We provide a `tfjs_bundle` macro to generate these bundles.

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
In the `src/BUILD.bazel` file, we compile the tests with `ts_library`. In the case of `tfjs-core`, we actually publish the test files, since other packages use them in their tests. Therefore, it's important that we set the `module_name` to `@tensorflow/tfjs-core/dist`. If a package's tests are not published, the `module_name` can probably be omitted. In a future major version of tfjs, we may stop publishing the tests to npm.

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

### Update the tests entrypoint
Many packages have a `src/run_tests.ts` file (or similar) that they use for selecting which tests to run. That file defines the paths to the test files that Jasmine uses. Since Bazel outputs appearin a different location, the paths to the test files must be updated. As an example, the following paths
```ts
const coreTests = 'node_modules/@tensorflow/tfjs-core/src/tests.ts';
const unitTests = 'src/**/*_test.ts';
```
would need to be updated to
```ts
const coreTests = 'tfjs-core/src/tests.js';
const unitTests = 'the-package-name/src/**/*_test.js';
```
Note that `.ts` has been changed to `.js`. This is because we're no longer running node tests with `ts-node`, so the input test files are now `.js` outputs created by the `ts_library` rule that compiled the tests.

It's also important to make sure the `nodejs_test` rule that runs the test has [`link_workspace_root = True`](https://bazelbuild.github.io/rules_nodejs/Built-ins.html#nodejs_binary-link_workspace_root). Otherwise, the test files will not be accessable at runtime.

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

The esbuild bundle is then used in the tfjs_web_test macro, which uses [karma_web_test](https://bazelbuild.github.io/rules_nodejs/Concatjs.html#karma_web_test) to serve it to a browser to be run. Different browserstack browsers can be enabled or disabled in the `browsers` argument, and the full list of browsers is located in `tools/karma_template.conf.js`. Browserstack browser tests are automatically tagged with `ci`.

`tfjs-core/BUILD.bazel`
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

Whereas before, tests were included based on the [`karma.conf.js`](https://github.com/tensorflow/tfjs/blob/2603aac08c5e41a9c6e5b79d294c6a61800acb67/tfjs-backend-webgl/karma.conf.js#L54-L58) file, now, tests must be included in the test bundle to be run. Make sure to `import` each test file in the test bundle's entrypoint. To help with this, we provide an `enumerate_tests` Bazel rule to generate a `tests.ts` file with the required imports.

[`tfjs-core/src/BUILD.bazel`](https://github.com/tensorflow/tfjs/blob/a32cc50dbd0dce2e6a53bb962eedc0d87a064b1e/tfjs-core/src/BUILD.bazel#L32-L48)
```starlark
load("//tools:enumerate_tests.bzl", "enumerate_tests")

# Generates the 'tests.ts' file that imports all test entrypoints.
enumerate_tests(
    name = "tests",
    srcs = [":all_test_entrypoints"], # all_test_entrypoints is a filegroup
    root_path = "tfjs-core/src",
)
```

### Update the package.json
1. Verify the entrypoints of the package.json to match the outputs generated by `tfjs_bundle` and `ts_library`. [tfjs-core/package.json](https://github.com/tensorflow/tfjs/blob/a32cc50dbd0dce2e6a53bb962eedc0d87a064b1e/tfjs-core/package.json#L6-L12) is an example.
    1. The main entrypoint should point to the node bundle, `dist/tf-package-name.node.js`.
    2. `jsnext:main` and `module` should point to the ESModule output `dist/index.js` created by `copy_ts_library_to_dist`.
2. If the package has browser tests, update the `sideEffects` field to include `.mjs` files generated by the `ts_library` under `./src` (e.g. `src/foo.mjs`). Bazel outputs directly to `src`, and although we copy those outputs to `dist` with another Bazel rule, the browser test bundles still import from `src`, so we need to mark them as sideEffects.

### Package for npm
We use the [pkg_npm](https://bazelbuild.github.io/rules_nodejs/Built-ins.html#pkg_npm) rule to create and publish the package to npm. However, there are a few steps needed before we can declare the package. For most packages, we distribute all our compiled outputs in the `dist` directory. However, due to how `ts_library` works, it creates outputs in the same directory as the source files were compiled from (except they show up in Bazel's `dist/bin` output dir). We need to copy these from `src` to `dist` while making sure Bazel is aware of this copy (so we can still use `pkg_npm`).

We also need to copy several other files to `dist`, such as the bundles created by `tfjs_bundle`, and we need to create [miniprogram](https://walkthechat.com/wechat-mini-programs-simple-introduction/) files for WeChat.

To copy files, we usually use the `copy_to_dist` rule. This rule creates symlinks to all the files in `srcs` and places them in a filetree with the same structure in `dest_dir` (which defaults to `dist`).

However, we can't just copy the output of a `ts_library`, since its default output is the `.d.ts` declaration files. We need to extract the desired ES Module `.mjs` outputs of the rule and rename them to have the `.js` extension. The `copy_ts_library_to_dist` does this rename, and it also copies the files to `dist` (including the `.d.ts` declaration files).

```starlark
load("//tools:copy_to_dist.bzl", "copy_ts_library_to_dist")

copy_ts_library_to_dist(
    name = "copy_src_to_dist",
    srcs = [
        "//tfjs-core/src:tfjs-core_lib",
        "//tfjs-core/src:tfjs-core_src_lib",
        "//tfjs-core/src:tfjs-core_test_lib",
    ],
    root = "src", # Consider 'src' to be the root directory of the copy
                  # (i.e. create 'dist/index.js' instead of 'dist/src/index.js')
    dest_dir = "dist", # Where to copy the files to. Defaults to 'dist', so it can
                       # actually be omitted in this case.
)
```

We can also copy the bundles output from `tfjs_bundle`

```starlark
copy_to_dist(
    name = "copy_bundles",
    srcs = [
        ":tf-core",
        ":tf-core.node",
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
        ":copy_test_snippets", # <- This is only in core, so I've omitted its definition.
    ],
)
```

Now the package can be published to npm with `bazel run //tfjs-core:tfjs-core_pkg.publish`.


### Configure Publishing to npm
With a `pkg_npm` rule defined, we add a script to `package.json` to run it. This script will be used by the main script that publishes the monorepo. 

```json
"scripts" {
    "publish-npm": "bazel run :tfjs-core_pkg.publish"
}
```

Since we now use the `publish-npm` script to publish this package instead of `npm publish`, we need to make sure the release tests and release script know how to publish it.

1. In `scripts/publish-npm.ts`, add your package's name to the `BAZEL_PACKAGES` set.
2. In `e2e/scripts/publish-tfjs-ci.sh`, add your package's name to the `BAZEL_PACKAGES` list.

You should also add a script to build the package itself without publishing (used for the `link-package`).

```json
"build": "bazel build :tfjs-core_pkg",
```

### Update Downstream `package.json` Paths

If no packages depend on your package (i.e. no `package.json` file includes your package via a `link` dependency), then you can skip this section.

As a core featue of its design, Bazel places outputs in a different directory than sources. Outputs are symlinked to `dist/bin/[package-name]/.....` instead of appearing in `[package-name]/dist`. Due to the different location, all downstream packages' `package.json` files need to be updated to point to the new outputs. However, due to some details of how Bazel and the Node module resolution algorithm work, we can't directly `link:` to Bazel's output.

Instead, we maintain a `link-package` pseudopackage where we copy the Bazel outputs. This package allows for correct Node module resolution between Bazel outputs because it has its own `node_modules` folder. This package will never be published and will be removed once the migration is complete.

#### Add the Package to `link-package`
Add your package to the `devDependencies` of the `link-package`'s `package.json`. The package path should be similar to the one below and is based off of the `pkg_npm` rule's name.

```json
"devDependencies": {
  "@tensorflow/tfjs-core": "file:../dist/bin/tfjs-core/tfjs-core_pkg",
  "@tensorflow/your-package": "file:../dist/bin/...",
}
```

#### Add a build script to the `link-package`
Add a script to build your package to the link-package's `package.json`. Be sure to add it to the `build` script as well.

```json
"scripts": {
  "build": "yarn build-backend-cpu && yarn build-core && yarn reinstall",
  "build-core": "cd ../tfjs-core && yarn && yarn build",
  "build-your-package": "...",
},
```

#### Add your package to the `reinstall` script
This ensures that when the link package is rebuilt, it uses the most up-to-date version of your package.

```json
    "reinstall": "yarn && yarn reinstall-link-package-core && yarn cache clean @tensorflow/your-package && ... && rimraf node_modules && yarn"
```

#### Change Downstream Dependency `package.json` Paths
Update all downstream dependencies that depend on the package to point to its location in the `link-package`.

```json
"devDependencies": {
  "@tensorflow/tfjs-core": "link:../link-package/node_modules/@tensorflow/tfjs-core",
  "@tensorflow/your-package": "link:../link-package/node_modules/@tensorflow/your-package",
},
```

To find downstream packages, run `grep -r --exclude=yarn.lock --exclude-dir=node_modules "link:.*your-package-name" .` in the root of the repository.

### Update or Remove `cloudbuild.yml`
Update the `cloudbuild.yml` to remove any steps that are now built with Bazel. These will be run by the `bazel-tests` step, which runs before other packages' steps. Any Bazel rule tagged as `ci` will be tested / build in CI.

Note that the output paths of Bazel-created outputs will be different, so any remaining steps that now rely on Bazel outputs may need to be updated. Bazel outputs are located in `tfjs/dist/bin/...`.

If all steps of the `cloudbuild.yml` file are handled by Bazel, it can be deleted. Make sure to also remove references to the package from `tfjs/scripts/package_dependencies.json`. This includes references to it from other steps in the dependency tree.

Rebuild the cloudbuild golden files by running `yarn update-cloudbuild-tests` in the root of the repository.

### Push to Git
Before pushing to Git, run the Bazel linter by running `yarn bazel:format` and `yarn bazel:lint-fix` in the root of the repo. We run the linter in CI, so if your build is failing in CI only, incorrectly formatted files may be the reason.

### Done!
🎉🎉🎉
