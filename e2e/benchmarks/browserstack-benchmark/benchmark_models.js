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

/**
 * The purpose of this test file is to benchmark models by a test runner, such
 * as karma. To invoke this test, inlude this file to the `files` field of
 * `karma.conf.js`.
 *
 * This file wraps the model benchmarking into a Jasmine test and the benchmark
 * results will be logged to the console.
 */

async function getEnvSummary() {
  let envSummary = `${tf.getBackend()} backend`;
  if (tf.getBackend() === 'webgl') {
    envSummary += `, version ${tf.env().get('WEBGL_VERSION')}`;
  } else if (tf.getBackend() === 'wasm') {
    const hasSIMD = await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT');
    envSummary += hasSIMD ? ' with SIMD' : ' without SIMD';
  }
  return envSummary;
}

async function getBenchmarkSummary(timeInfo, memoryInfo, modelName = 'model') {
  if (timeInfo == null) {
    throw new Error('Missing the timeInfo parameter.');
  } else if (timeInfo.times.length === 0) {
    throw new Error('Missing the memoryInfo parameter.');
  } else if (memoryInfo == null) {
    throw new Error('The length of timeInfo.times is at least 1.');
  }

  const numRuns = timeInfo.times.length;
  const envSummary = await getEnvSummary();
  const benchmarkSummary = `
  benchmark the ${modelName} on ${envSummary}
  1st inference time: ${printTime(timeInfo.times[0])}
  Average inference time (${numRuns} runs): ${printTime(timeInfo.averageTime)}
  Best inference time: ${printTime(timeInfo.minTime)}
  Peak memory: ${printMemory(memoryInfo.peakBytes)}
  `;
  return benchmarkSummary;
}

describe('benchmark models', () => {
  beforeAll(() => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it(`benchmark ${benchmarks[benchmark_parameters.model]}`, async () => {
    try {
      await tf.setBackend(benchmark_parameters.backend);

      // Load the model.
      const benchmark = benchmarks[benchmark_parameters.model];
      const numRuns = benchmark_parameters.numRuns;
      let model;
      if (benchmark_parameters.model === 'custom') {
        if (benchmark_parameters.modelUrl == null) {
          throw new Error('Please provide model url for the custom model.')
        }
        model = await loadModelByUrl(benchmark_parameters.modelUrl);
      } else {
        model = await benchmark.load();
      }

      // Benchmark.
      let timeInfo;
      let memoryInfo;
      if (benchmark.predictFunc != null) {
        const predict = benchmark.predictFunc();
        timeInfo = await profileInferenceTime(() => predict(model), numRuns);
        memoryInfo = await profileInferenceMemory(() => predict(model));
      } else {
        const input = generateInput(model);
        timeInfo = await profileInferenceTimeForModel(model, input, numRuns);
        memoryInfo = await profileInferenceMemoryForModel(model, input);
      }

      // Report results.
      const resultStr =
          `<benchmark>${JSON.stringify({timeInfo, memoryInfo})}</benchmark>`;
      console.log(resultStr);
    } catch (error) {
      console.log(`<error>${error}</error>`);
    }
  });
});
