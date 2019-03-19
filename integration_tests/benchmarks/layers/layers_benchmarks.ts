/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';

describe('foo', () => {
  const DATA_SERVER_ROOT = 'http://localhost:8090/';
  const BENCHMARKS_JSON_URL = `${DATA_SERVER_ROOT}/benchmarks.json`;

  async function getBenchmarkModelNames(): Promise<string[]> {
    const benchmarks = await (await fetch(BENCHMARKS_JSON_URL)).json();
    return benchmarks.models as string[];
  }

  async function loadModel(modelName: string): Promise<tf.Model> {
    const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/model.json`;
    const modelJSON = await (await fetch(modelJSONPath)).json();
    return await tf.models.modelFromJSON(modelJSON['modelTopology']);
  }

  async function loadModelBenchmarkData(modelName: string): Promise<any> {
    const modelJSONPath = `${DATA_SERVER_ROOT}/${modelName}/data.json`;
    return await (await fetch(modelJSONPath)).json();
  }

  it('Benchmark models', async () => {
    const modelNames = await getBenchmarkModelNames();
    
    for (let i = 0; i < modelNames.length; ++i) {
      const modelName = modelNames[i];
      console.log(`Benchmarking ${modelName}`);
      const model = await loadModel(modelName);
      model.summary();
      const benchmarkData = await loadModelBenchmarkData(modelName);
      console.log(benchmarkData);
    }
  });
});
