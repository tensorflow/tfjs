# tensorflowjs: The Python Package for TensorFlow.js

The **tensorflowjs** pip package contains libraries and tools for
[TensorFlow.js](https://js.tensorflow.org).

Use following command to install the library with support of interactive CLI:
```bash
pip install tensorflowjs[wizard]
```

Then, run the following to see a list of CLI options

```bash
tensorflowjs_converter --help
```

or, use the wizard

```bash
tensorflowjs_wizard
```

Alternatively, run the converter via its Bazel target. This must be run from withing the tfjs repo:

```bash
yarn bazel run //tfjs-converter/python/tensorflowjs/converters:converter -- --help
```

## Development

The python tests are run with Bazel.

```bash
yarn bazel test //tfjs-converter/python/...
```

Alternatively, run `yarn run-python-tests` to run the above command.

To debug a specific test case, use the `--test_filter` option. For example,

```bash
yarn bazel test //tfjs-converter/python/tensorflowjs/converters:tf_saved_model_conversion_v2_test --test_filter=ConvertTest.test_convert_saved_model_v1
```

Interactive debugging with breakpoints is supported by `debugpy` in VSCode.
To enable debugging, put this code at the top of the test file you want to
debug.

```python
import debugpy
debugpy.listen(('localhost', 5724))
print("Waiting for debugger to connect. See tfjs-converter python README")
debugpy.wait_for_client()
```

You may also need to add the following dependency to the test target in the
Bazel `BUILD` file if it's not already present.
```starlark
"//tfjs-converter/python/tensorflowjs:expect_debugpy_installed"
```

Then, run the test with `bazel run --config=debugpy` and connect
the VSCode debugger by selecting the `Python: Attach (Converter)` option.
