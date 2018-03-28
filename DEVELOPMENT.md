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
  yarn lint
  ```
* As a required step for code review and submission, run and pass all unit
  tests:
  ```bash
  yarn test
  ```
  The above command opens Chrome and Firefox and uses them to run all TypeScript
  unit tests.

## Code Structure

## Types

* Tensors are represented as `Tensor` from TensorFlow.js Core.
* Tensor shape is represented as an array of number (`number[]`), in a way
  consistent with TensorFlow.js Core.
