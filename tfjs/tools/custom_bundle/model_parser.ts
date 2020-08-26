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

import {esmModuleProvider} from './esm_module_provider';

export function getOps(modelJson: io.ModelArtifacts): string[] {
  const kernel2op = kernelToOpMapping();
  const results: Set<string> = new Set();

  const graph = modelJson.modelTopology as tensorflow.IGraphDef;
  const nodes = graph.node;
  nodes.forEach((node) => {
    const ops = kernel2op[node.op];
    if (ops == null) {
      console.log(`Kernel => Op warning: could not find op mapping for kernel ${
          node.op}`);
    }
    ops.forEach((op: string) => results.add(op));
  });
  return Array.from(results);
}

function kernelToOpMapping() {
  const mappingPath = esmModuleProvider.pathToKernel2OpMapping();
  return JSON.parse(fs.readFileSync(mappingPath, 'utf-8'));
}
