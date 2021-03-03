/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 *  This file is used to load a saved model and perform inference.
 *  Run this script in console:
 *   ts-node index.ts --model_path=MODEL_PATH --inputs_dir=INPUTS_DIR
 *   --outputs_dir=OUTPUTS_DIR
 *
 *  For help, run:
 *   ts-node inference.ts -h
 */

import '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-cpu';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import * as path from 'path';
import * as yargs from 'yargs';

import {FileHandler} from './file_handler';

// Placeholder for g3 import.

// Following cmd options casing tradition.
// tslint:disable-next-line:enforce-name-casing
interface Options {
  model_path: string;
  inputs_dir: string;
  outputs_dir: string;
  inputs_data_file: string;
  inputs_shape_file: string;
  inputs_dtype_file: string;
  tf_input_name_file: string;
  tf_output_name_file: string;
  backend: string;
}
// tslint:enable:enforce-name-casing

/**
 *   The entry point of tfjs inference api.
 */
async function main() {
  const argParser = yargs.options({
    model_path: {
      description: 'Path to the tfjs model json file.',
      type: 'string',
      demandOption: true
    },
    inputs_dir: {
      description: 'Directory to read the input tensor info and output ' +
          'info files.',
      type: 'string',
      demandOption: true
    },
    outputs_dir: {
      description:
          'Directory to write the output files. Output files include: ' +
          'data.json, shape.json and dtype.json. The order of the output ' +
          'tensors follow the same order as the tf_output_name_file. If the ' +
          'file is not provided, the default outputs of the model would be ' +
          'used.',
      type: 'string',
      demandOption: true
    },
    inputs_data_file: {
      description: 'Filename of the input data file. Must provide the file.',
      type: 'string',
      default: 'data.json'
    },
    inputs_shape_file: {
      description: 'Filename of the input shape file. Must provide the file.',
      type: 'string',
      default: 'shape.json'
    },
    inputs_dtype_file: {
      description: 'Filename of the input dtype file. Must provide the file.',
      type: 'string',
      default: 'dtype.json'
    },
    tf_input_name_file: {
      description: 'Filename of the input name of the tf model. The input ' +
          'names should match the names defined in the signatureDef of the ' +
          'model. Must provide the file.',
      type: 'string',
      default: 'tf_input_name.json'
    },
    tf_output_name_file: {
      description: 'Filename of the output name of the tf model. The output ' +
          'names should match the names defined in the signatureDef of the ' +
          'model. If the file is not provided, the default outputs of the ' +
          'model would be used.',
      type: 'string'
    },
    backend: {
      description: 'Choose which tfjs backend to use. Supported backends: ' +
          'cpu|wasm',
      type: 'string',
      default: 'cpu'
    }
  });

  const options = argParser.argv as {} as Options;

  if (options.backend === 'wasm') {
    // Placeholder for g3 specific code.
    await tfc.setBackend('wasm');
  } else if (options.backend === 'cpu') {
    await tfc.setBackend('cpu');
  } else {
    throw new Error(
        'Only cpu and wasm backend is supported, but got ' + options.backend);
  }

  const model =
      await tfconv.loadGraphModel(new FileHandler(options.model_path));

  // Read in input tensor info and output info, then convert to json.
  const inputsDataString = fs.readFileSync(
      path.join(options.inputs_dir, options.inputs_data_file), 'utf8');
  const inputsData = JSON.parse(inputsDataString);

  const inputsShapeString = fs.readFileSync(
      path.join(options.inputs_dir, options.inputs_shape_file), 'utf8');
  const inputsShape = JSON.parse(inputsShapeString);

  const inputsDtypeString = fs.readFileSync(
      path.join(options.inputs_dir, options.inputs_dtype_file), 'utf8');
  const inputsDtype = JSON.parse(inputsDtypeString);

  const tfInputNameString = fs.readFileSync(
      path.join(options.inputs_dir, options.tf_input_name_file), 'utf8');
  const inputName = JSON.parse(tfInputNameString);

  let outputName;
  if (options.tf_output_name_file) {
    const tfOutputNameString = fs.readFileSync(
        path.join(options.inputs_dir, options.tf_output_name_file), 'utf8');
    outputName = JSON.parse(tfOutputNameString);
  }

  const namedInputs =
      createInputTensors(inputsData, inputsShape, inputsDtype, inputName);

  const result = await model.executeAsync(namedInputs, outputName);

  // executeAsync can return a single tensor or an
  // array of tensors. We wrap the single tensor in an array so that later
  // operation can always assume to operate on an iterable result.
  const ys = (model.outputs.length === 1 ? [result] : result) as tfc.Tensor[];

  // Write out results to file.
  const ysData = [];
  const ysShape = [];
  const ysDtype = [];
  for (let i = 0; i < ys.length; i++) {
    const y = ys[i];
    ysData.push(await y.data());
    ysShape.push(y.shape);
    ysDtype.push(y.dtype);
  }

  fs.writeFileSync(
      path.join(options.outputs_dir, 'data.json'), JSON.stringify(ysData));
  fs.writeFileSync(
      path.join(options.outputs_dir, 'shape.json'), JSON.stringify(ysShape));
  fs.writeFileSync(
      path.join(options.outputs_dir, 'dtype.json'), JSON.stringify(ysDtype));

  // Dispose all tensors.
  Object.keys(namedInputs).forEach(key => namedInputs[key].dispose());
  ys.forEach(tensor => tensor.dispose());
}

/**
 * Create a list of input tensors.
 *
 * @private
 * @param inputsData An array with each element being the value to
 *    create a tensor.
 * @param inputsShape An array with each element being
 *    the shape to create a tensor.
 * @param inputsDtype An array with each element being the
 *    dtype to create a tensor.
 * @param inputName An array of input names, identifies the
 *    input tensor in the same order as the other input arrays.
 * @return An array of tensors.
 */
function createInputTensors(
    inputsData: tfc.TypedArray[], inputsShape: number[][],
    inputsDtype: tfc.DataType[], inputName: string[]): tfc.NamedTensorMap {
  const xs: tfc.Tensor[] = [];
  for (let i = 0; i < inputsData.length; i++) {
    const input = tfc.tensor(inputsData[i], inputsShape[i], inputsDtype[i]);
    xs.push(input);
  }

  return inputName.reduce((map: tfc.NamedTensorMap, name, index) => {
    map[name] = xs[index];
    return map;
  }, {});
}

main();
