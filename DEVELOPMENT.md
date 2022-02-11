## Development

This repository is a monorepo that contains the following NPM packages:

APIs:
- [TensorFlow.js Core](/tfjs-core),
  a flexible low-level API for neural networks and numerical computation.
- [TensorFlow.js Layers](/tfjs-layers),
  a high-level API which implements functionality similar to
  [Keras](https://keras.io/).
- [TensorFlow.js Data](/tfjs-data),
  a simple API to load and prepare data analogous to
  [tf.data](https://www.tensorflow.org/guide/datasets).
- [TensorFlow.js Converter](/tfjs-converter),
  tools to import a TensorFlow SavedModel to TensorFlow.js
- [TensorFlow.js Vis](/tfjs-vis),
  in-browser visualization for TensorFlow.js models
- [TensorFlow.js AutoML](/tfjs-automl),
  Set of APIs to load and run models produced by
  [AutoML Edge](https://cloud.google.com/vision/automl/docs/edge-quickstart).


Backends/Platforms:
- [TensorFlow.js CPU Backend](/tfjs-backend-cpu), pure-JS backend for Node.js and the browser.
- [TensorFlow.js WebGL Backend](/tfjs-backend-webgl), WebGL backend for the browser.
- [TensorFlow.js WASM Backend](/tfjs-backend-wasm), WebAssembly backend for the browser.
- [TensorFlow.js WebGPU](/tfjs-backend-webgpu), WebGPU backend for the browser.
- [TensorFlow.js Node](/tfjs-node), Node.js platform via TensorFlow C++ adapter.
- [TensorFlow.js React Native](/tfjs-react-native), React Native platform via expo-gl adapter.

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
This will install yarn dependencies, build the other TensorFlow.js packages that the package being tested depeds on,
and run the tests for the package. During development, you may want to run `yarn test-dev` instead to avoid
unnecessarily rebuilding dependencies.

Many TensorFlow.js packages use Karma to run tests in a browser. These tests can be configured by command-line options.

To run a subset of tests:

```bash
$ yarn test --//:grep=multinomial
 
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

#### Developing on Windows
Developing on Windows is supported through the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) running Debian.

1. Install WSL2 (if necessary) by following [Microsoft's instructions](https://docs.microsoft.com/en-us/windows/wsl/install). WSL1 has not been tested, but it may work.
2. Install Debian in WSL2. [Debian is available from the Microsoft store](https://www.microsoft.com/en-us/p/debian/9msvkqc78pk6?activetab=pivot:overviewtab).
3. Open Debian and install node and yarn with 
`sudo apt update && sudo apt install nodejs && npm i -g yarn`. 
If you need to reset the root debian password, you can get a root shell from command prompt with `wsl -u root`.
4. Make sure Chrome is installed on Windows. Then, find the path to `chrome.exe`. It's probably `C:\Program Files\Google\Chrome\Application\chrome.exe`.
5. Run the following to set up the `CHROME_BIN` variable, clone the `tfjs` repo, and create a custom `.bazelrc.user` config for WSL. If your `chrome.exe` is not located at the above path, you will need to change it to the correct path in the command below.
```bash
# Add yarn bin to the path
echo "export PATH=$PATH:~/.yarn/bin/" >> ~/.bashrc &&
# Set CHROME_BIN. Change this if your CHROME_BIN has a different path.
echo "export CHROME_BIN=/mnt/c/Program\ Files/Google/Chrome/Application/chrome.exe" >> ~/.bashrc &&
source ~/.bashrc &&
# Clone tfjs.
git clone https://github.com/tensorflow/tfjs.git &&
cd tfjs &&
# Create the .bazelrc.user file for WSL.
echo "# Pass necessary WSL variables for running in Windows Subsystem for Linux.
# WSLENV and WSL_DISTRO_NAME are build-in variables that are needed for running
# the 'wslpath' command, which Karma uses to resolve file paths.
# DISPLAY=:0 is passed to the Chrome process to make it launch in a window
# since running Chrome headlessly from WSL does not seem to work. If you get
# this working, please send a PR updating these docs (or open an issue :).
run --test_env=CHROME_BIN --test_env=WSLENV --test_env=WSL_DISTRO_NAME --define DISPLAY=:0
test --test_env=CHROME_BIN --test_env=WSLENV --test_env=WSL_DISTRO_NAME --define DISPLAY=:0" > .bazelrc.user &&
printf "\n\nDone! Try running a browser test to verify the installation worked, e.g. 'cd tfjs-core && yarn && yarn test-browser'\n"
```
6. To access this repo from VScode, follow [Microsoft's WSL VSCode Tutorial](https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode).

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
