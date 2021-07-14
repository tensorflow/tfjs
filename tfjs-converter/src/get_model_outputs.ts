// Copyright 2021 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Use the CPU backend.
import '@tensorflow/tfjs-backend-cpu';
import * as tfc from '@tensorflow/tfjs-core';

import * as argparse from 'argparse';
import * as fs from 'fs';
import * as path from 'path';

import {loadGraphModel} from './executor/graph_model';

const parser = new argparse.ArgumentParser();
parser.addArgument('--model_dir', {
  required: true,
  nargs: 1,
  help: 'Directory where TFJS converted model artifacts are found'
});

parser.addArgument('--output_nodes_path', {
  required: true,
  nargs: 1,
  help: 'Path to file containing output nodes to run model on'
});

parser.addArgument('--input_values', {
  nargs: '*',
  help:
      'Inputs to the graph as JSON values (given in sorted order of input name)'
});

parser.addArgument(
    '--output_file',
    {required: true, nargs: 1, help: 'Path where output tensors are stored'});

async function main() {
  const args = parser.parseArgs();
  const modelDir = args.model_dir[0] as string;
  const modelDirFiles = fs.readdirSync(modelDir);
  const outputNodeNames = fs.readFileSync(args.output_nodes_path[0], 'utf-8')
                              .split('\n')
                              .filter(Boolean);
  const outputFilePath = args.output_file[0];

  if (modelDirFiles.indexOf('model.json') === -1) {
    throw new Error('Model path directory must contain a model.json file');
  }

  const weightPaths =
      modelDirFiles.filter(fileName => fileName.endsWith('.bin'));

  const modelFile = fs.readFileSync(path.join(modelDir, 'model.json'), 'utf-8');
  const pathToWeightsData: {[fileName: string]: ArrayBuffer} = {};
  weightPaths.forEach(
      fileName => pathToWeightsData[fileName] =
          fs.readFileSync(path.join(modelDir, fileName)).buffer);

  const model =
      await loadGraphModel(tfc.io.rawFiles(modelFile, pathToWeightsData));
  const modelNodes = model.nodes;
  const modelNodeNames = modelNodes.reduce<Set<string>>((nodeNames, node) => {
    nodeNames.add(node.name);
    return nodeNames;
  }, new Set());
  // Remove nodes not found in the model, or not asked for by script caller
  const desiredNodeNames = outputNodeNames.filter(
      nodeName => modelNodeNames.has(nodeName.split(':')[0]));

  const outputValues: {[nodeName: string]: [number[], number[]]} = {};
  const inputValues =
      (args.input_values as string[])
          .map(inputValue => tfc.tensor(JSON.parse(inputValue)));
  try {
    const nodeOutputValues =
        await model.executeAsync(inputValues, desiredNodeNames) as tfc.Tensor[];
    for (let i = 0; i < desiredNodeNames.length; ++i) {
      const nodeOutputValue = nodeOutputValues[i];
      outputValues[desiredNodeNames[i]] =
          [Array.from(await nodeOutputValue.data()), nodeOutputValue.shape];
    }
  } catch (e) {
    console.log('Error running model on all nodes: ', e);
    console.log('Switching to running node by node instead');
    for (const nodeName of desiredNodeNames) {
      try {
        const nodeOutputValue =
            await model.executeAsync(inputValues, nodeName) as tfc.Tensor;
        outputValues[nodeName] =
            [Array.from(await nodeOutputValue.data()), nodeOutputValue.shape];
      } catch (e) {
        outputValues[nodeName] = null;
      }
    }
  }

  fs.writeFileSync(outputFilePath, JSON.stringify(outputValues));

  process.exit(0);
}

main();
