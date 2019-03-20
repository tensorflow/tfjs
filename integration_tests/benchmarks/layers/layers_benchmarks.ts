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

// import * as math from 'mathjs';
import * as tf from '@tensorflow/tfjs';

import {ModelTaskLog} from '../types';

function getRandomInputsAndOutputs(model: tf.Model, batchSize: number):
    {xs: tf.Tensor|tf.Tensor[], ys: tf.Tensor|tf.Tensor[]} {
  return tf.tidy(() => {
    let xs: tf.Tensor|tf.Tensor[] = [];
    for (const input of model.inputs) {
      xs.push(tf.randomUniform([batchSize].concat(input.shape.slice(1))));
    }
    if (xs.length === 1) {
      xs = xs[0];
    }

    let ys: tf.Tensor|tf.Tensor[] = [];
    for (const output of model.outputs) {
      ys.push(tf.randomUniform([batchSize].concat(output.shape.slice(1))));
    }
    if (ys.length === 1) {
      ys = ys[0];
    }

    return {xs, ys};
  });
}

describe('TF.js Layers Benchmarks', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 600000;
  });

  // Karma serves static files under the base/ path.
  const DATA_SERVER_ROOT = './base/data';
  const BENCHMARKS_JSON_URL = `${DATA_SERVER_ROOT}/benchmarks.json`;

  async function getBenchmarkModelNamesAndConfig(): Promise<{
    modelNames: string[],
    predictNumBurnInRuns: number,
    predictNumBenchmarkRuns: number
  }> {
    const benchmarks = await (await fetch(BENCHMARKS_JSON_URL)).json();
    console.log(JSON.stringify(benchmarks));  // DEBUG
    return {
      modelNames: benchmarks.models as string[],
      predictNumBurnInRuns: benchmarks.config.PREDICT_BURNINS as number,
      predictNumBenchmarkRuns: benchmarks.config.PREDICT_RUNS as number
    };
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
    const {modelNames, predictNumBenchmarkRuns, predictNumBurnInRuns} =
        await getBenchmarkModelNamesAndConfig();

    for (let i = 10; i < modelNames.length; ++i) {
      const modelName = modelNames[i];
      console.log(`Benchmarking ${modelName}`);
      const model = await loadModel(modelName);
      const benchmarkData = await loadModelBenchmarkData(modelName);

      const batchSize = benchmarkData.batch_size as number;

      const {xs, ys} = getRandomInputsAndOutputs(model, batchSize);

      // Burn-in runs.
      let predictOut: tf.Tensor|tf.Tensor[];
      for (let i = 0; i < predictNumBurnInRuns; ++i) {
        predictOut = model.predict(xs);
      }
      if (Array.isArray(predictOut)) {
        for (const out of predictOut) {
          await out.data();
        }
      } else {
        await predictOut.data();
      }
      tf.dispose(predictOut);

      // Benchmarked runs.
      // const ts: number[] = [];
      const t0 = performance.now();
      for (let i = 0; i < predictNumBenchmarkRuns; ++i) {
        predictOut = model.predict(xs);
        // ts.push(performance.now() - t0);
        // tf.dispose(predictOut);
      }
      if (Array.isArray(predictOut)) {
        for (const out of predictOut) {
          await out.data();
        }
      } else {
        await predictOut.data();
      }
      tf.dispose(predictOut);
      const t1 = performance.now();

      const taskLog: ModelTaskLog = {
        numBurnInRuns: predictNumBurnInRuns,
        numBenchmarksRuns: predictNumBenchmarkRuns,
        batchSize,
        averageTimeMs: (t1 - t0) / predictNumBenchmarkRuns,
        medianTimeMs: NaN,
        minTimeMs: NaN,
        // averageTimeMs: math.mean(ts),
        // medianTimeMs: math.median(ts),
        // minTimeMs: math.min(ts),
        timestamp: new Date().getTime()
      };
      console.log(taskLog);

      tf.dispose({xs, ys});
    }
  });
});
