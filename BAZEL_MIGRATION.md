# This doc is WIP
# Bazel Migration

This document details the steps to migrate a package to build with Bazel. These steps are easiest to understand with a working example, so this doc references `tfjs-core`'s setup as much as possible. Since this migration is still in the early phases, the steps and processes listed here may change as we improve on the process, add features to each package's build, and create tfjs-specific build functions.

## Scope
Migrating a package to Bazel involves adding Bazel targets that build the package, run its tests, and pack the package for publishing to npm. To ensure a seamless transition to Bazel, we're maintaining the original build system and building the Bazel build system in parallel with it. __When transitioning a package to build with Bazel, it's important not to break the current build system used for that package.__

## Caveats
- Bazel will only make a dependency or file available to the build if you explicitly declare it as a dependency / input to the rule you're using.
- All Bazel builds use the root package.json, so you may have to add packages to it. As long as you use `@npm//dependency-name` in BUILD files to add dependencies, you won't need to worry about the build accidentally seeing the package's `node_modules` directory instead of the root `node_modules`. Bazel will only make the root `node_modules` directory visible to the build.
  - Even though the build doesn't use the package's `node_modules`, you may have to run `yarn` within the package to get code completion to work correctly. I'm looking into why this is the case.
- There may be issues with depending on explicitly pinned versions of `@tensorflow` scoped packages, which might affect some demos if they're migrated to use Bazel. We might just want to leave demos out of Bazel so they're easier to understand.


## Steps
These steps are general guidelines for how to build a package with Bazel. They should work for most packages, but there may be some exceptions (e.g. wasm, react native).

### Make sure all dependencies build with Bazel
A package's dependencies must be migrated before it can be migrated.

### Create a `BUILD.bazel` file in the package
Bazel looks for targets to run in `BUILD` and `BUILD.bazel` files. Use the `.bazel` extension since blaze uses `BUILD`. You may want to install an extension for your editor to get syntax highlighting. [Here's the vscode extension](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel).

### Compile the package with `ts_project`
[ts_project](https://bazelbuild.github.io/rules_nodejs/TypeScript.html#ts_project) compiles the package. It runs `tsc --project` on typescript source files.

Here's an example of how `tfjs-core` uses `ts_project` to build.

```starlark
load("@npm//@bazel/typescript:index.bzl", "ts_project")

TEST_SRCS = ["src/setup_test.ts"]
ts_project(
    name = "tfjs-core_lib",           # Other targets reference this target using this name
    srcs = glob(                      # .ts sources to compile
        ["src/**/*.ts"],
        exclude = TEST_SRCS,
    ),
    declaration = True,               # From tsconfig.json
    extends = "//:tsconfig.json",     # The package's tsconfig extends the root tsconfig
    incremental = True,               # From tsconfig.json
    out_dir = "dist",                 # From tsconfig.json
    root_dir = "src",                 # From tsconfig.json
    source_map = True,                # From tsconfig.json
    tsconfig = "tsconfig.json",       # Use the package's tsconfig
    deps = [                          # Other targets this library depends on, including npm packages
        "@npm//@types",               # and other tfjs packages. See tfjs-backend-cpu/BUILD.bazel for
        "@npm//jasmine-core",         # an example of including other tfjs packages.
        "@npm//seedrandom",
    ],
)
```
### Bundle the package
This step involves bundling the compiled files from the compilation step into a single file. TFJS generates several bundles for each package in order to support different execution environments. At the moment, the instructions here do not create es5 compatible bundles. These instructions will change in the future, and it's likely that we will have a single `tfjs_bundle` function that can be imported to create all the necessary bundles instead of having to write them all out separately. At the moment, however, you'll need to use the [esbuild rule](https://bazelbuild.github.io/rules_nodejs/esbuild.html) for each bundle listed in `package.json`.

```starlark
load("//:esbuild.bzl", "esbuild")

esbuild(
    name = "tf-core.min",            # This should match package.json. The output file has '.js' appended automatically.
    entry_point = "dist/index.js",   # This should match the 'input' field in the package's rollup file
                                     # except the path will be to the compiled .js file instead of the .ts file.
    external = [                     # These are external to the bundle. They can be copied from the rollup file.
        "node-fetch",
        "util",
    ],
    minify = True,
    sources_content = True,          # Tell esbuild to include source file content in the generated sourcemaps instead
                                     # of referencing files. Sourcemaps are broken without this. This does not increase
                                     # bundle size since it only affects the sourcemaps.
    deps = [
        ":tfjs-core_lib",
    ],
)

```

### Declare the package as importable
In order to make `import {something} from '@tensorflow/tfjs-core'` work when we use `tfjs-core` in other packages, we need to declare the package using `js_library`. This is not very well documented at the moment, and may eventually be replaced with `pkg_npm`. See [this article](https://hackmd.io/@alexeagle/SkNE_w2QU), [this section of documentation](https://bazelbuild.github.io/rules_nodejs/Built-ins.html#npm_install-generate_local_modules_build_files), and [this issue](https://github.com/bazelbuild/rules_nodejs/issues/149) for more info.
```starlark
load("@build_bazel_rules_nodejs//:index.bzl", "js_library")

js_library(
    name = "tfjs-core",
    package_name = "@tensorflow/tfjs-core",  # The name that can be imported in other .ts files
    srcs = [
        "package.json",
    ],
    deps = [
        ":tf-core.min",                      # Include all bundles.
        ":tf-core.node",
        ":tfjs-core_lib",                    # ...and include the compiled files.
    ],
)
```


### Compile the tests with `ts_library`
Since (in most cases) the tests are not published, we don't have to worry about where their files end up when compiled. We can use [ts_library](https://bazelbuild.github.io/rules_nodejs/TypeScript.html#ts_library-1) to compile them.

```starlark
load("@npm//@bazel/typescript:index.bzl", "ts_library")

ts_library(
    name = "tfjs-core_test_lib",
    testonly = True,                           # This rule can only be used by tests.
    srcs = [                                   # .ts files to compile. In most cases, this would be 'TEST_SRCS' as defined
        "setup_test_bazel.ts",                 # above, but tfjs-core is a special case and couldn't use the current
    ],                                         # 'setup_test.ts' file. We include a separate 'setup_test_bazel.ts' file instead.
    tsconfig = "//:tsconfig_ts_library.json",  # Use a different tsconfig because ts_library doesn't support 'incremental'.
    deps = [
        ":tfjs-core",
        "@npm//@types",
        "@npm//jasmine-core",
        "@npm//seedrandom",
    ],
)
```

### Run node tests with jasmine_node_test
[jasmine_node_test](https://bazelbuild.github.io/rules_nodejs/Jasmine.html#jasmine_node_test) can run Jasmine tests that run in node. There aren't actually any node tests added to the repo yet (so if you add one, please update this section with an example :).

### Run browser tests with karma_web_test
Like the section on bundling, this section will likely be changed to use a single custom `tfjs_web_test` rule instead of several different rules to set up tests. Right now, we use `esbuild` to bundle the tests into a single file and [karma_web_test](https://bazelbuild.github.io/rules_nodejs/Concatjs.html#karma_web_test) to serve them to a browser to be run.

```starlark
esbuild(
    name = "tfjs-core_test_bundle",
    testonly = True,
    entry_point = "setup_test_bazel.ts",
    external = [
        # webworker tests call 'require('@tensorflow/tfjs')', which              
        # is external to the test bundle.                                        
        # Note: This is not a bazel target. It's just a string.                  
        "@tensorflow/tfjs",
    ],
    sources_content = True,
    deps = [
        ":tfjs-core_lib",
        ":tfjs-core_test_lib",
        "//tfjs-backend-cpu",
    ],
)

karma_web_test(
    name = "tfjs-core_test",
    srcs = [
        ":tfjs-core_test_bundle",
    ],
    static_files = [
        # Listed here so sourcemaps are served                                   
        ":tfjs-core_test_bundle",
    ],
    tags = ["native"],
)
```
