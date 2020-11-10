## Development

This repository contains only the logic and scripts that combine the following NPM packages:
- TensorFlow.js Core (@tensorflow/tfjs-core),
  a flexible low-level linear algebra and gradients API.
- TensorFlow.js Layers (@tensorflow/tfjs-layers),
  a high-level model-centric API which implements functionality similar to
  [Keras](https://keras.io/).
- TensorFlow.js Converter (@tensorflow/tfjs-converter),
  a library to load converted TensorFlow SavedModels and execute them in JavaScript.
- TensorFlow.js Data (@tensorflow/tfjs-data),
  a library to load and prepare data for use in machine learning models.

#### Yarn
We use yarn, and if you are adding or removing dependencies you should use yarn
to keep the `yarn.lock` file up to date.

#### Code editor
We recommend using [Visual Studio Code](https://code.visualstudio.com/) for
development. Make sure to install
[TSLint VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-typescript-tslint-plugin)
and the npm [clang-format](https://github.com/angular/clang-format) `1.2.2` or later
with the
[Clang-Format VSCode extension](https://marketplace.visualstudio.com/items?itemName=xaver.clang-format)
for auto-formatting.

#### Testing
Before submitting a pull request, make sure the code passes all the tests and is clean of lint errors:

```bash
# cd into the package directory you want to test
$ yarn test
$ yarn lint
```

To run a subset of tests and/or on a specific browser:

```bash
$ yarn test --browsers=Chrome --grep='multinomial'
 
> ...
> Chrome 62.0.3202 (Mac OS X 10.12.6): Executed 28 of 1891 (skipped 1863) SUCCESS (6.914 secs / 0.634 secs)
```

To run the tests once and exit the karma process (helpful on Windows):

```bash
$ yarn test --single-run
```

To run the tests in an environment that does not have GPU support (such as Chrome Remote Desktop):

```bash
$ yarn test --testEnv cpu
```

Available test environments: cpu, webgl1, webgl2.

#### Packaging (browser and npm)

In any of the directories the following commands build the NPM tarball:

```bash
$ yarn build-npm
> Stored standalone library at dist/tf-core(.min).js
> Stored also tensorflow-tf-core-VERSION.tgz
```

To install it locally, run `yarn add ./tensorflow-tf-core-VERSION.tgz`.

> On Windows, use bash (available through git) to use the scripts above.

Looking to contribute, and don't know where to start? Check out our "stat:contributions welcome" [issues](https://github.com/tensorflow/tfjs/labels/stat%3Acontributions%20welcome).


## For repository owners: commit style guide

When merging commits into master, it is important to follow a few conventions
so that we can automatically generate release notes and have a uniform commit
history.

1. When you squash and merge, the default commit body will be all of the
commits on your development branch (not the PR description). These are usually
not very useful, so you should remove them, or replace them with the PR
description.

2. Release notes are automatically generated from commits. We have introduced a
few tags which help sort commits into categories for release notes:

- FEATURE (when new functionality / API is added)
- BREAKING (when there is API breakage)
- BUG (bug fixes)
- PERF (performance improvements)
- DEV (development flow changes)
- DOC (documentation changes)
- SECURITY (security changes)

These tags correspond to GitHub labels which are automatically prepended to your PR description.
Please add the appropriate labels to your PR.

A typical commit may look something like:

```
Subject: Add tf.toPixels. (#900)
Body:
FEATURE

tf.toPixels is the inverse of tf.fromPixels, writing a tensor to a canvas.

```

This will show up under "Features" as:
- Add tf.toPixels. (#900). Thanks, @externalcontributor.


You can also use multiple tags for the same commit if you want it to show up in
two sections. You can add clarifying text on the line of the tags.

You can add clarifying messages on the line of the tag as well.

For example:

```
Subject: Improvements to matMul. (#900)
Body:

FEATURE Add transpose bits to matmul.
PERFORMANCE Improve matMul CPU speed by 100%.
```

This will show up under "Features" as:
- Add transpose bits to matmul (Improvements to matMul.) (#900). Thanks, @externalcontributor.

This will also show up under "Performance" as:
- Improve matMul CPU speed by 100%. (Improvements to matMul.) (#900). Thanks, @externalcontributor.
To build **TensorFlow.js Core API** from source, we need to clone the project and prepare
the dev environment:

```bash
$ git clone https://github.com/tensorflow/tfjs-core.git
$ cd tfjs-core
$ yarn # Installs dependencies.
```