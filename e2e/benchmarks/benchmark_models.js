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

async function getEnvInfo() {
  let envInfo = `${tf.getBackend()} backend`;
  if (tf.getBackend() === 'webgl') {
    envInfo += `, version ${tf.env().get('WEBGL_VERSION')}`;
  } else if (tf.getBackend() === 'wasm') {
    const hasSIMD = await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT')
    envInfo += hasSIMD ? ' with SIMD' : ' without SIMD';
  }
  return envInfo;
}

describe('benchmark models', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it('mobile net', async () => {
    const url =
        'https://storage.googleapis.com/learnjs-data/mobilenet_v2_100_fused/model.json';
    const model = await tf.loadGraphModel(url);
    const input = generateInput(model);
    const predict = () => model.predict(input);

    const numRuns = 20;
    const times = await profileInferenceTime(predict, numRuns);
    const memory = await profileInferenceMemory(predict);
    const averageTime =
        times.reduce((acc, curr) => acc + curr, 0) / times.length;
    const minTime = Math.min(...times);

    let benchmarkInfo = 'benchmark mobilenet_v2 on ';
    benchmarkInfo += await getEnvInfo();
    console.log(benchmarkInfo);
    console.log('1st inference time', printTime(times[0]));
    console.log(
        `Average inference time (${numRuns} runs)`, printTime(averageTime));
    console.log('Best inference time', printTime(minTime));
    console.log('Peak memory', printMemory(memory.peakBytes));
  });
});
