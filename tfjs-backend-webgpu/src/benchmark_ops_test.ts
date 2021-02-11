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

import * as tf from '@tensorflow/tfjs-core';
import {describeWebGPU} from './test_util';

describeWebGPU('Ops benchmarks', () => {
  // Performs `trials` trials, of `reps` repetitions each. At the end of each
  // trial, endTrial() is run (and included in the benchmark time). This
  // allows the cost of endTrial() to be amortized across the many iterations.
  // This is needed in particular because WebGPU readbacks are asynchronous
  // and therefore always incur latency. (Plus, in Chrome right now, readbacks
  // are very inefficient, making the problem way worse.) Readbacks could be
  // avoided by using fences, but we don't have a common abstraction over
  // WebGL and WebGPU fences at the moment.
  async function time(
      doRep: (r: number) => tf.Tensor[] | tf.Tensor,
      endTrial?: () => Promise<void>, disposeAfterEachTrial = false,
      trials = 50, reps = 1) {
    const times = [];

    let toDispose: tf.Tensor[] = [];
    const dispose = () => {
      for (const t of toDispose) {
        t.dispose();
      }
      toDispose = [];
    };

    const trial = async () => {
      let result;
      for (let r = 0; r < reps; ++r) {
        result = doRep(r);

        toDispose = toDispose.concat(Array.isArray(result) ? result : [result]);
      }

      if (endTrial != null) {
        await endTrial();
      } else {
        await (Array.isArray(result) ? result[0] : result).data();
      }
    };

    // Warm-up. Specifically, this pre-allocates enough memory for an entire
    // trial, ensuring that no allocations happen when timing a trial (if the
    // backend reuses allocations).
    await trial();
    dispose();

    for (let t = 0; t < trials; ++t) {
      const start = tf.util.now();
      await trial();
      times.push(tf.util.now() - start);
      if (disposeAfterEachTrial) {
        dispose();
      }
    }

    const mean = times.reduce((a, b) => a + b, 0) / trials;
    const min = Math.min(...times);
    const fmt = (n: number) => n.toFixed(3);
    console.log(`Mean time: ${fmt(mean)} ms -> ${fmt(mean / reps)} / rep`);
    console.log(`Min time: ${fmt(min)} ms -> ${fmt(min / reps)} / rep`);
  }

  it('argMax', async () => {
    const n = 2;
    const doTest = async (axis: number) => {
      const tensors = new Array(n);
      const maxes = new Array(n);
      for (let i = 0; i < n; ++i) {
        tensors[i] = tf.randomNormal([100, 100, 100]);
      }

      await time(
          (r) => {
            maxes[r] = tf.argMax(tensors[r], axis);
            return [];
          },
          async () => {
            await maxes[maxes.length - 1].data();
            for (const t of maxes) {
              t.dispose();
            }
          },
          false, 50, n);
    };

    await doTest(0);
    await doTest(1);
    await doTest(2);
  }, 60000);

  it('concat', async () => {
    const a = tf.randomNormal([500, 500]);
    const b = tf.randomNormal([500, 500]);

    await time(() => tf.concat([a, b], 1));
  });

  it('resizeBilinear', async () => {
    const input = tf.randomNormal<tf.Rank.R3>([128, 128, 4]);

    await time(() => input.resizeBilinear([256, 256], false));
  });

  it('matMul', async () => {
    const a = tf.randomNormal([500, 500]);
    const b = tf.randomNormal([500, 500]);

    await time(() => tf.matMul(a, b));
  });

  it('matMul - dispatch 1', async () => {
    const a = tf.randomNormal([16, 2048]);
    const b = tf.randomNormal([2048, 16]);

    await time(() => tf.matMul(a, b));
  });

  it('add', async () => {
    const a = tf.randomNormal([1, 65, 65, 256]);
    const b = tf.randomNormal([1, 65, 65, 256]);

    await time(() => tf.add(a, b));
  });

  it('clip', async () => {
    const a = tf.randomNormal([1, 65, 65, 256]);

    await time(() => tf.clipByValue(a, 0.1, 0.9));
  });

  it('conv2d', async () => {
    const a = tf.randomNormal<tf.Rank.R4>([1, 128, 128, 4]);
    const b = tf.randomNormal<tf.Rank.R4>([25, 25, 4, 4]);

    await time(() => tf.conv2d(a, b, 1, 'same'));
  });

  it('conv2dWithInChannel3', async () => {
    const a = tf.randomNormal<tf.Rank.R4>([1, 231, 231, 3]);
    const b = tf.randomNormal<tf.Rank.R4>([7, 7, 3, 64]);

    await time(() => tf.conv2d(a, b, 2, 'valid'));
  });

  it('depthwiseconv2d', async () => {
    const x = tf.randomNormal<tf.Rank.R4>([1, 128, 128, 1]);
    const w = tf.tensor4d(
        [0.303873, 0.229223, 0.144333, 0.803373],
        [2, 2, 1, 1],
    );

    await time(() => tf.depthwiseConv2d(x, w, 1, 'valid'));
  });

  it('maxPool with filter size = 1', async () => {
    const y = tf.randomNormal<tf.Rank.R4>([1, 57, 57, 256]);
    const z = tf.randomNormal<tf.Rank.R4>([1, 29, 29, 512]);
    await time(() => tf.maxPool(y, 1, 2, 'same'), null, true, 10, 10);
    await time(() => tf.maxPool(z, 1, 2, 'same'), null, true, 10, 10);
  });

  it('maxPool', async () => {
    const x = tf.randomNormal<tf.Rank.R4>([1, 131, 131, 64]);

    await time(() => tf.maxPool(x, 2, 1, 'same'), null, true, 10, 10);
  });

  it('prelu', async () => {
    const x = tf.randomNormal([500]);
    const a = tf.randomNormal([500]);

    await time(() => tf.prelu(x, a), null, false, 10, 10);
  });

  it('slice', async () => {
    const a = tf.randomNormal<tf.Rank.R1>([500]);

    await time(() => tf.slice1d(a, 2, 498), null, false, 10, 10);
  });

  it('transpose', async () => {
    const x = tf.randomNormal([1024, 1024]);
    await time(() => tf.transpose(x, [1, 0]), null, false, 10, 10);
  });

  it('stridedSlice', async () => {
    const a = tf.randomNormal<tf.Rank.R1>([500]);

    await time(() => tf.stridedSlice(a, [0], [500], [2]), null, true, 10, 10);
  });
});
