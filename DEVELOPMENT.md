# Notes for TensorFlow.js Layers Developers

## Build and Test

* As a preparatory step, run:
  ```bash
  yarn
  ```
* Our TypeScript source code follows the Google
  [clang format](https://clang.llvm.org/docs/ClangFormatStyleOptions.html).
  A script is available to automatically format .ts files:
  `tools/clang_format_ts.sh`.
  Be sure to run the script before you proceed to the next step.
  To use this script on all .ts files under `src/`, do:
  ```bash
  tools/clang_format_ts.sh -a
  ```
  To use this script on .ts files touched in the current git workspace, simply
  do:
  ```bash
  tools/clang_format_ts.sh
  ```
  See the doc string of the script for other modes of usage. If necessary,
  clang-format can be installed with:
  ```bash
  apt-get install clang-format
  ```
  Note that it is not sufficient to use the clang format in Visual Studio Code,
  because its results differ from those of the `clang-format` command and
  `tools/clang_format_ts.sh`.
* As a required step for code review and submission, run and pass the TypeScript
  linter (`tslint`):
  ```bash
  yarn run lint
  ```
* As a required step for code review and submission, run and pass all unit
  tests:
  ```bash
  yarn run test
  ```
  The above command opens Chrome and Firefox and uses them to run all TypeScript
  unit tests.

### Python Development

There are some Python libraries, binary and tests in the `scripts/` directory.

As a prerequisite, install the following dependencies for python testing
* `pip install h5py`
* `pip install keras`
* `pip install numpy`
* `pip install tensorflow`

For Python linter, install `pylint`, e.g.,
* `apt-get install -y pylint`

To run the Python linter:
```sh
cd python
pylint tensorflowjs
```

To run the python unit tests, there are two options. You can choose the one that
you prefer.

1. Run the tests using the `run-python-tests.sh` script:

   ```sh
   cd python
   ./run-python-tests.sh
   ```

2. Run the tests using Bazel. See bazel installation guide
   [here](https://docs.bazel.build/versions/master/install.html). Once bazel
   is installed, do:

   ```sh
   cd python
   ./copy_write_weights.sh
   bazel test scripts/...
   ```

Be sure to run the tests under **both** Python 2 and Python 3.

#### Building the tensorflowjs pip package

```sh
cd python

# You need to specify a folder where the pip wheel file will be stored, e.g.,
./build-pip-package.sh /tmp/my_tensorflowjs_pip

# If the script succeeds, you canuse `pip install` to install the pip package:

pip install --force-reinstall \
  /tmp/my_tensorflowjs_pip/tensorflowjs-0.0.1-py2-none-any.whl
```

## Code Structure

## Types

* Tensors are represented as `Tensor` from deeplearn.js.
* Tensor shape is represented as an array of number (`number[]`), in a way
  consistent with deeplearn.js.
