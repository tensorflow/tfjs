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
  Subsequent average inference time (${numRuns} runs): ${printTime(timeInfo.averageTimeExclFirst)}
  Best inference time: ${printTime(timeInfo.minTime)}
  Peak memory: ${printMemory(memoryInfo.peakBytes)}
  `;
  return benchmarkSummary;
}

const KARMA_SERVER = './base';

async function benchmarkModel(benchmarkParameters) {
  // Load the model.
  const benchmark = benchmarks[benchmarkParameters.model];
  const numRuns = benchmarkParameters.numRuns;
  let model;
  if (benchmarkParameters.model === 'custom') {
    if (benchmarkParameters.modelUrl == null) {
      throw new Error('Please provide model url for the custom model.');
    }
    model = await loadModelByUrl(benchmarkParameters.modelUrl);
  } else {
    model = await benchmark.load();
  }

  // Benchmark.
  let timeInfo;
  let memoryInfo;
  if (benchmark.predictFunc != null) {
    const predict = benchmark.predictFunc();
    timeInfo = await timeInference(() => predict(model), numRuns);
    memoryInfo = await profileInference(() => predict(model));
  } else {
    const input = generateInput(model);
    timeInfo = await timeModelInference(model, input, numRuns);
    memoryInfo = await profileModelInference(model, input);
  }

  return { timeInfo, memoryInfo };
}

async function benchmarkCodeSnippet(benchmarkParameters) {
  let predict = null;

  const setupCodeSnippetEnv = benchmarkParameters.setupCodeSnippetEnv || '';
  const codeSnippet = benchmarkParameters.codeSnippet || ''
  eval(setupCodeSnippetEnv.concat(codeSnippet));

  if (predict == null) {
    throw new Error(
      'predict function is suppoed to be defined in codeSnippet.');
  }

  // Warm up.
  await timeInference(predict, 1);

  // Benchmark code snippet.
  timeInfo = await timeInference(predict, benchmarkParameters.numRuns);
  memoryInfo = await profileInference(predict);

  return { timeInfo, memoryInfo };
}

describe('BrowserStack benchmark', () => {
  let benchmarkParameters;
  beforeAll(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
    const response = await fetch(`${KARMA_SERVER}/benchmark_parameters.json`);
    benchmarkParameters = await response.json();
    tf.env().set('SOFTWARE_WEBGL_ENABLED', true);
  });

  it(`benchmark`, async () => {
    try {
      // Setup benchmark environments.
      const targetBackend = benchmarkParameters.backend;
      await tf.setBackend(targetBackend);

      // Run benchmark and stringify results.
      let resultObj;
      if (benchmarkParameters.model === 'codeSnippet') {
        resultObj = await benchmarkCodeSnippet(benchmarkParameters);
      } else {
        resultObj = await benchmarkModel(benchmarkParameters);
      }

      // Get GPU hardware info.
      resultObj.gpuInfo =
        targetBackend === 'webgl' ? (await getRendererInfo()) : 'MISS';

      // Report results.
      console.log(
        `<tfjs_benchmark>${JSON.stringify(resultObj)}</tfjs_benchmark>`);
    } catch (error) {
      console.log(`<tfjs_error>${error}</tfjs_error>`);
    }
  });
});
