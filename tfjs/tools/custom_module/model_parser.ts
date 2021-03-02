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

// tslint:disable-next-line: no-imports-from-dist
import * as tensorflow from '@tensorflow/tfjs-converter/dist/data/compiled_api';
import {io} from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import {CustomTFJSBundleConfig} from './types';
import {bail} from './util';

export function getOpsForConfig(
    // tslint:disable-next-line: no-any
    config: CustomTFJSBundleConfig, kernelToOps: any) {
  // This will return a list of ops used by the model.json(s) passed in.
  const results: Set<string> = new Set();
  let modelJson;
  for (const modelJsonPath of config.models) {
    try {
      modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf-8'));
    } catch (e) {
      bail(`Error loading JSON file ${modelJsonPath}`);
    }

    const ops = getOps(modelJson, kernelToOps);
    ops.forEach((op: string) => results.add(op));
  }
  return Array.from(results);
}

export function getOps(
    // tslint:disable-next-line: no-any
    modelJson: io.ModelArtifacts, kernelToOp: any): string[] {
  const results: Set<string> = new Set();

  const addOpsToResults = (kernel: string) => {
    const ops = kernelToOp[kernel];
    if (ops == null) {
      console.warn(
          `Kernel => Op warning: could not find op mapping for kernel ${
              kernel}`);
    }
    ops.forEach((op: string) => {
      results.add(op);
    });
  };

  const graph = modelJson.modelTopology as tensorflow.IGraphDef;

  // Parse nodes
  if (graph.node != null) {
    graph.node.forEach((node) => {
      addOpsToResults(node.op);
    });
  }

  // Parse functionDef nodes
  if (graph.library != null && graph.library.function != null) {
    graph.library.function.forEach((functionDef) => {
      const nodeDef = functionDef.nodeDef;
      if (nodeDef != null) {
        nodeDef.forEach((node) => {
          addOpsToResults(node.op);
        });
      }
    });
  }

  return Array.from(results);
}
