# TensorFlow.js Inference API

This package provides a cli tool for model inferencing in a Node env.
Additionally, the package is compiled to a binary with the same functionality,
which allows to use tfjs for ML tasks in any envs and platforms.

The tool can be used to validate TFJS model execution against python results.

**Note**: This package is under development.

## How to use

### Run the cli tool in Node env.
Checkout the code: `git clone https://github.com/tensorflow/tfjs.git`

Go to tfjs-inference directory: `cd tfjs-inference`

Install dependencies: `yarn`

Run the inference:
```
ts-node src/index.ts --model_path=MODEL_PATH --inputs_dir=INPUTS_DIR
 *   --outputs_dir=OUTPUTS_DIR
```
The script expects three required arguments: `model_path`, `inputs_dir` and
`outputs_dir`. There are also optional arguments, see the options below.

**Options**

**model_path**: Directory to a tfjs model json file.

**inputs_dir**: Directory to read the input tensor info and output info files.

**outputs_dir**: Directory to write the output files. Output files include:
                 data.json, shape.json and dtype.json. Additionally, name.json
                 is written if the model returns a map. The order of the output
                 tensors follow the same order as the tf_output_name_file.

**inputs_data_file**: (Optional) Filename of the input data file.
                      Default to data.json.

**inputs_shape_file**: (Optional) Filename of the input shape file.
                       Default to shape.json

**inputs_dtype_file**: (Optional) Filename of the input dtype file.
                       Default to dtype.json

**tf_input_name_file**: (Optional) Filename of the input name of the tf model.
                        The input names should match the names defined in the
                        signatureDef of the model.
                        Default to tf_input_name.json

**tf_output_name_file**: (Optional) Filename of the output name of the tf model.
                         The output names should match the names defined in
                         the signatureDef of the model.
                         Default to tf_output_name.json

**backend**: Choose which tfjs backend to use. Supported backends: cpu|wasm.
             Default to cpu.

**Notes**:
* The program requires these files to exist in the inputs_dir:
  `inputs_data_file`, `inputs_shape_file`, `inputs_dtype_file`, and
  `tf_input_name_file`.
* For `model_path`, absolute path is preferred. It also supports relative path,
  which should be relative to tfjs-inference directory.
* About the input and output formats. They are represented as array of tensors.
  The data.json, shape.json, dtype.json and tf_input_name.json files together
  represent the array of tensors. [Example](https://github.com/tensorflow/tfjs/tree/master/tfjs-inference/test_data)
