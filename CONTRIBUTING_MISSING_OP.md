## How to Contribute a New op to TF.js Repository

### Step 1. Add the new op to tfjs-core ops directory.
**Prerequisite**
1. The op should already be supported in [TF](https://www.tensorflow.org/api_docs/python/tf/all_symbols) api.
2. The op should not have been supported in TF.js, see [op support list](https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/docs/supported_ops.md).
3. Check [here](https://docs.google.com/spreadsheets/d/1D25XtWaBrmUEErbGQB0QmNhH-xtwHo9LDl59w0TbxrI/edit#gid=0) for various ops supported in different backends. 

**Implementation Details**
1. Create a new op in `tfjs-core/ops` directory.

    The op file should have the following elements:
        1. License information.
        2. JSDoc comment.
        3. Input validations, if any.
        4. Delegate the execution to the right kernel through `ENGINE.runKernel()`.

    In addition, for the kernel delegation to work properly, in `kernel_names.ts`
  file, define:
        1. kernel name.
        2. input type.
        3. attribute type.

2. Export the op file in ops.ts in the same directory.
3. Add tests for the op in the same directory. Test file name should be the same as the op file’s name with _test suffix.
4. Exclude the test in all the backends, and add annotation “Not implemented yet.”. See below for where to exclude the test in each backend:

**cpu backend**

In `run_tests.ts`, `customInclude` method, add:
```
    // Not implemented yet.
    if (testName.includes(test_name)) {
      return false;
    }
```

**webgl backend**

In `setup_test.ts`, `customInclude` method, add:
```
    // Not implemented yet.
    if (testName.includes(test_name)) {
      return false;
    }
```

**node backend**

In `run_tests.ts`, `IGNORE_LIST`, add test_name to the list.

**Example PRs**
* [CPU and WebGL backend SparseReshape](https://github.com/tensorflow/tfjs/pull/4956)

### Step 2. Add a new kernel in a backend.
**Implementation Details**
1. Create a new kernel in the `kernels` directory of a backend.

    The kernel file should have the following elements:
    1. License information.
    2. The kernel implementation, and export it.
    3. Export a `kernelConfig`.

2. Register the kernel in `register_all_kernels.ts` in the corresponding backend.

3. Remove the op from test exclusion list in the corresponding backend. For wasm
   backend, add the test to the inclusion list.
   
**Example PRs**
* [CPU and WebGL backend SparseReshape](https://github.com/tensorflow/tfjs/pull/4956)
* [WASM backend Round](https://github.com/tensorflow/tfjs/pull/4486)

### Step 3. Add the op to Converter’s executor.
1. Add op mapping in the op list ts file, use your best judgement to assign an op category: `tfjs-converter/python/tensorflowjs/op_list/{corresponding_op_category}.json`. The corresponding `.ts` file will be automatically generated when converter is built. Use
the [TF C++ op api](https://www.tensorflow.org/api_docs/cc/) as reference for tfOpName, inputs, and attrs.

2. Find the corresponding executor for the op and add the op to the switch, the [executors](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter/src/operations/executors) are in `tfjs-converter/src/operations/executors`.

3. Add a test to the corresponding executor test file.

4. Update the supported op doc in `tfjs-converter/docs/supported_ops.md`.

**Example PRs**
* [SparseReshape Op](https://github.com/tensorflow/tfjs/pull/4963)
