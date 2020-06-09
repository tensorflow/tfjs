/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 *   ts-node inference.ts --model_path=MODEL_PATH -inputs_dir=INPUTS_DIR
 *   -outputs_dir=OUTPUTS_DIR
 *
 *  For help, run:
 *   ts-node inference.ts -h
 */

import '@tensorflow/tfjs-backend-wasm';

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tfc from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {join} from 'path';
import * as yargs from 'yargs';

import {FileHandler} from './file_handler';

// Following cmd options casing tradition.
// tslint:disable-next-line:enforce-name-casing
interface Options {
  model_path: string;
  inputs_dir: string;
  outputs_dir: string;
  inputs_data_file: string;
  inputs_shape_file: string;
  inputs_dtype_file: string;
}
// tslint:enable:enforce-name-casing

/**
 *   The entry point of tfjs inference api.
 */
async function main() {
  const argParser = yargs.options({
    model_path: {
      description: 'Directory to a tfjs model json file.',
      type: 'string',
      demandOption: true
    },
    inputs_dir: {
      description: 'Directory to read the model json files.',
      type: 'string',
      demandOption: true
    },
    outputs_dir: {
      description:
          'Directory to write the output files. Output files include: data.json, shape.json and dtype.json.',
      type: 'string',
      demandOption: true
    },
    inputs_data_file: {
      description: 'Filename of the input data file.',
      type: 'string',
      default: 'data.json'
    },
    inputs_shape_file: {
      description: 'Filename of the input shape file.',
      type: 'string',
      default: 'shape.json'
    },
    inputs_dtype_file: {
      description: 'Filename of the input dtype file.',
      type: 'string',
      default: 'dtype.json'
    }
  });

  const options = argParser.argv as {} as Options;

  await tfc.setBackend('wasm');

  const model =
      await tfconv.loadGraphModel(new FileHandler(options.model_path));

  // Read in input files.
  const inputsDataString = fs.readFileSync(
      join(options.inputs_dir, options.inputs_data_file), 'utf8');
  const inputsData = JSON.parse(inputsDataString);
  const inputsShapeString = fs.readFileSync(
      join(options.inputs_dir, options.inputs_shape_file), 'utf8');
  const inputsShape = JSON.parse(inputsShapeString);
  const inputsDtypeString = fs.readFileSync(
      join(options.inputs_dir, options.inputs_dtype_file), 'utf8');
  const inputsDtype = JSON.parse(inputsDtypeString);

  const xs = createInputTensors(inputsData, inputsShape, inputsDtype);

  const result = await model.executeAsync(xs);

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
      join(options.outputs_dir, 'data.json'), JSON.stringify(ysData));
  fs.writeFileSync(
      join(options.outputs_dir, 'shape.json'), JSON.stringify(ysShape));
  fs.writeFileSync(
      join(options.outputs_dir, 'dtype.json'), JSON.stringify(ysDtype));
}

/**
 * Create a list of input tensors.
 *
 * @private
 * @param inputsData An array with each element being the value to
 *    create a tensor.
 * @param inputsShapes An array with each element being
 *    the shape to create a tensor.
 * @param inputsDtype An array with each element being the
 *    dtype to create a tensor.
 * @return An array of tensors.
 */
function createInputTensors(
    inputsData: tfc.TypedArray[], inputsShapes: number[][],
    inputsDtype: tfc.DataType[]) {
  const xs = [];
  for (let i = 0; i < inputsData.length; i++) {
    const input = tfc.tensor(inputsData[i], inputsShapes[i], inputsDtype[i]);
    xs.push(input);
  }

  return xs;
}

main();
