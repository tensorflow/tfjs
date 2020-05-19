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

import * as tfc from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {describeWebGPU} from './test_util';

describeWebGPU('Models benchmarks', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  async function benchmark(
      benchmarkFn: () => tf.Tensor | tf.Tensor[] |
          {[name: string]: tf.Tensor}): Promise<number> {
    const start = performance.now();
    const result = benchmarkFn();

    if (result instanceof Array) {
      await result[0].data();
    } else if (result instanceof tf.Tensor) {
      const tmp = result;
      await result.data();
      tmp.dispose();
    } else {
      await result;
    }
    return performance.now() - start;
  }

  it('resnet50', async () => {
    const input = tf.randomNormal([1, 224, 224, 3]);
    const url =
        'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/quant2/model-stride32.json';
    const model = await tfc.loadGraphModel(url);
    const bench = () => model.predict(input);
    await benchmark(bench);
    const times = [];
    const trials = 50;
    for (let t = 0; t < trials; ++t) {
      const time = await benchmark(bench);
      times.push(time);
    }
    input.dispose();
    console.log(times);
    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n: number) => n.toFixed(3);
    console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / 1)} / rep`);
    console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / 1)} / rep`);
  });
});
