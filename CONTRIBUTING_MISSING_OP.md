## How to Contribute a New op to TF.js Repository

### Step 1. Add the new op to tfjs-core ops directory.
**Prerequisite**
1. The op should already be supported in [TF](https://www.tensorflow.org/api_docs/python/tf/all_symbols) api.
2. The op should not have been supported in TF.js.

**Implementation Details**
1. Create a new op in `tfjs-core/ops` directory.

  The op file should have below elements:
    1. License information.
    2. Doccomment.
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

### Step 2. Add a new kernel in a backend.
**Implementation Details**
1. Create a new kernel in the `kernels` directory of a backend.

  The kernel file should have below elements:
  1. License information.
  2. The kernel implementation, and export it.
  3. Export a `kernelConfig`.

2. Register the kernel in `register_all_kernels.ts` in the corresponding backend.

### Step 3. Add the op to Converter’s executor.
1. Add op mapping in the op list json files, use your best judgement to assign an op category: `tfjs-converter/src/operations/op_list/{corresponding_op_category}.ts`

2. Auto generates corresponding json file by running following command in directory tfjs-converter: `yarn gen-json`. Check that the json object is generated in `python/tensorflowjs/op_list/{corresponding_op_group}.json`

3. Find the corresponding executor for the op and add the op to the switch, the executors are in `tfjs-converter/src/operations/executors`.

4. Update the supported op doc in `tfjs-converter/docs/supported_ops.md`.
