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
// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags, TestEnv} from '@tensorflow/tfjs-core/dist/jasmine_util';

export function describeWebGPU(name: string, tests: (env: TestEnv) => void) {
  describeWithFlags('webgpu ' + name, ALL_ENVS, tests);
}

declare global {
  interface Window {
    records: any[];
    testingBackend: string;
  }
}

window.records = [];

// Performs `trials` trials, of `reps` repetitions each. At the end of each
// trial, endTrial() is run (and included in the benchmark time). This
// allows the cost of endTrial() to be amortized across the many iterations.
export async function benchmark(
    name: string, doRep: (r: number) => any, endTrial?: () => Promise<void>,
    disposeAfterEachTrial = false, trials = 20, reps = 20) {
  const times = [];

  let toDispose: tf.Tensor[] = [];
  const dispose = () => {
    for (const t of toDispose) {
      if (typeof t.dispose === 'function') {
        t.dispose();
      }
    }
    toDispose = [];
  };

  const trial = async () => {
    let result;
    for (let r = 0; r < reps; ++r) {
      result = await doRep(r);

      toDispose = toDispose.concat(Array.isArray(result) ? result : [result]);
    }

    if (endTrial != null) {
      await endTrial();
    } else {
      if (Array.isArray(result)) {
        await result[0].data();
      } else {
        if (typeof result.data === 'function') {
          await result.data();
        }
      }
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

  dispose();

  const mean = times.reduce((a, b) => a + b, 0) / trials;
  const min = Math.min(...times);
  const fmt = (n: number) => n.toFixed(3);
  console.log(`${name}: ${fmt(mean)} / ${fmt(min)}`, tf.getBackend());

  const record = {
    name,
    mean: fmt(mean),
    min: fmt(min),
    numTrials: trials,
    backend: tf.getBackend()
  };
  window.records.push(record);
  window.testingBackend = tf.getBackend();
  return;
}

export async function benchmarkAndLog(
    name: string, doRep: (r: number) => any, endTrial?: () => Promise<void>,
    disposeAfterEachTrial = false, trials = 20, reps = 20) {
  await benchmark(name, doRep, endTrial, disposeAfterEachTrial, trials, reps);
  tf.setBackend('webgpu');
  await benchmark(name, doRep, endTrial, disposeAfterEachTrial, trials, reps);
}
