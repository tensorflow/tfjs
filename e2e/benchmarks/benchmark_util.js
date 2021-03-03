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
 * This tool depends on tf-core, tf-layers, tf-converter and the backends
 * (tf-backend-cpu, tf-backend-webgl or tf-backend-wasm) that you would use.
 */

/**
 * Generates a random input for `model`, based on `model.inputs`. For
 * tf.GraphModel, `NamedTensorMap` input will be returned; otherwise,
 * `Tensor[]` will be returned.
 *
 * ```js
 * const model = tf.sequential(
 *    {layers: [tf.layers.dense({units: 1, inputShape: [3]})]});
 * const input = generateInput(model);
 * const prediction = await model.predict(input);
 *
 * console.log(`Generated input: ${Object.values(input)}`);
 * console.log(`Prediction for the generated input: ${prediction}`);
 * ```
 *
 * @param model The model object that is used to generated the input.
 */
function generateInput(model) {
  if (model == null) {
    throw new Error('The model does not exist.');
  } else if (model.inputs == null) {
    throw new Error('The model.inputs cannot be found.');
  }

  const inputDefs = model.inputs.map((inputNode, inputNodeIndex) => {
    // Replace -1 or null in input tensor shape.
    const inputShape = inputNode.shape.map(shapeValue => {
      if (shapeValue == null || shapeValue < 0) {
        return 1;
      } else {
        return shapeValue;
      }
    });
    return {
      shape: inputShape,
      name: inputNode.name,
      dtype: inputNode.dtype,
      range: [0, 1000]
    };
  });

  return generateInputFromDef(inputDefs, model instanceof tf.GraphModel);
}

/**
 * Generates a random input for input definition.
 *
 * ```js
 * const input = generateInput(inputDefs);
 *
 * console.log(`Generated input: ${Object.values(input)}`);
 * console.log(`Prediction for the generated input: ${prediction}`);
 * ```
 *
 * @param inputDefs The input definition that is used to generate the input.
 * @param isForGraphModel flag for whether to generate inputs for GraphModel
 */
function generateInputFromDef(inputDefs, isForGraphModel = false) {
  if (inputDefs == null) {
    throw new Error('The inputDef cannot be found.');
  }

  const tensorArray = [];
  try {
    inputDefs.forEach((inputDef, inputDefIndex) => {
      const inputShape = inputDef.shape;

      // Construct the input tensor.
      let inputTensor;
      if (inputDef.dtype === 'float32' || inputDef.dtype === 'int32') {
        // We assume a bell curve normal distribution. In this case,
        // we use below approximation:
        // mean ~= (min + max) / 2
        // std ~= (max - min) / 4
        // Note: for std, our approximation is based on the fact that
        // 95% of the data is within the range of 2 stds above and
        // below the mean. So 95% of the data falls in the range of
        // 4 stds.
        const min = inputDef.range[0];
        const max = inputDef.range[1];
        const mean = (min + max) / 2;
        const std = (max - min) / 4;
        generatedRaw = tf.randomNormal(inputShape, mean, std, inputDef.dtype);
        // We clip the value to be within [min, max], because 5% of
        // the data generated maybe outside of [min, max].
        inputTensor = tf.clipByValue(generatedRaw, min, max);
        generatedRaw.dispose();
      } else {
        throw new Error(
            `The ${inputDef.dtype} dtype of '${inputDef.name}' input ` +
            `at model.inputs[${inputDefIndex}] is not supported.`);
      }
      tensorArray.push(inputTensor);
    });

    // Return tensor map for tf.GraphModel.
    if (isForGraphModel) {
      const tensorMap = inputDefs.reduce((map, inputDef, i) => {
        map[inputDef.name] = tensorArray[i];
        return map;
      }, {});
      return tensorMap;
    }

    return tensorArray;
  } catch (e) {
    // Dispose all input tensors when the input construction is failed.
    tensorArray.forEach(tensor => {
      if (tensor instanceof tf.Tensor) {
        tensor.dispose();
      }
    });
    throw e;
  }
}

/**
 * Wrap the model's predict function (`model.predict` for tf.LayersModel
 * and `model.executeAsync` for tf.GraphModel) with the input.
 *
 * @param model An instance of tf.GraphModel or tf.LayersModel for finding and
 *     wrapping the predict function.
 * @param input The input tensor container for model inference.
 */
function getPredictFnForModel(model, input) {
  let predict;
  if (model instanceof tf.GraphModel) {
    // Because there's no straightforward way to analyze whether a graph has
    // dynamic op, so we try to use `execute` and, if it fails, we will fall
    // back to `executeAsync`.
    try {
      tf.tidy(() => {
        model.execute(input);
      });
      predict = () => model.execute(input);
    } catch (e) {
      predict = async () => await model.executeAsync(input);
    }
  } else if (model instanceof tf.LayersModel) {
    predict = () => model.predict(input);
  } else {
    throw new Error(
        'Predict function was not found. Please provide a tf.GraphModel or ' +
        'tf.LayersModel');
  }
  return predict;
}

/**
 * Executes the predict function for `model` (`model.predict` for tf.LayersModel
 * and `model.executeAsync` for tf.GraphModel) and times the inference process
 * for `numRuns` rounds. Then returns a promise that resolves with information
 * about the model's inference time:
 * - `times`: an array of inference time for each inference
 * - `averageTime`: the average time of all inferences
 * - `minTime`: the minimum time of all inferences
 * - `maxTime`: the maximum time of all inferences
 *
 * The inference time contains the time spent by both `predict()` and `data()`
 * called by tensors in the prediction.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const timeInfo =
 *    await timeModelInference(model, zeros, 2);
 *
 * console.log(`Elapsed time array: ${timeInfo.times}`);
 * console.log(`Average time: ${timeInfo.averageTime}`);
 * console.log(`Minimum time: ${timeInfo.minTime}`);
 * console.log(`Maximum time: ${timeInfo.maxTime}`);
 * ```
 *
 * @param model An instance of tf.GraphModel or tf.LayersModel for timing the
 *     inference process.
 * @param input The input tensor container for model inference.
 * @param numRuns The number of rounds for timing the inference process.
 */
async function timeModelInference(model, input, numRuns = 1) {
  const predict = getPredictFnForModel(model, input);
  return timeInference(predict, numRuns);
}

/**
 * Executes `predict()` and times the inference process for `numRuns` rounds.
 * Then returns a promise that resolves with information about the inference
 * time:
 * - `times`: an array of inference time for each inference
 * - `averageTime`: the average time of all inferences
 * - `minTime`: the minimum time of all inferences
 * - `maxTime`: the maximum time of all inferences
 *
 * The inference time contains the time spent by both `predict()` and `data()`
 * called by tensors in the prediction.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const timeInfo =
 *    await timeInference(() => model.predict(zeros), 2);
 *
 * console.log(`Elapsed time array: ${timeInfo.times}`);
 * console.log(`Average time: ${timeInfo.averageTime}`);
 * console.log(`Minimum time: ${timeInfo.minTime}`);
 * console.log(`Maximum time: ${timeInfo.maxTime}`);
 * ```
 *
 * @param predict The predict function to execute and time.
 * @param numRuns The number of rounds for `predict` to execute and time.
 */
async function timeInference(predict, numRuns = 1) {
  if (typeof predict !== 'function') {
    throw new Error(
        'The first parameter should be a function, while ' +
        `a(n) ${typeof predict} is found.`);
  }

  const times = [];
  for (let i = 0; i < numRuns; i++) {
    const start = performance.now();
    const res = await predict();
    // The prediction can be tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}.
    const value = await downloadValuesFromTensorContainer(res);
    const elapsedTime = performance.now() - start;

    tf.dispose(res);
    times.push(elapsedTime);
  }

  const averageTime = times.reduce((acc, curr) => acc + curr, 0) / times.length;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const timeInfo = {
    times,
    averageTime,
    minTime,
    maxTime

  };
  return timeInfo;
}

/**
 * Downloads the values from the `tensorContainer` from any `tf.Tensor`s found
 * within the `tensorContainer`. Returns a promise of `TypedArray` or
 * `TypedArray[]` that resolves when the computation has finished.
 *
 * The values are asynchronously downloaded in parallel.
 *
 * @param tensorContainer The container of tensors to be downloaded.
 */
async function downloadValuesFromTensorContainer(tensorContainer) {
  let valueContainer;
  if (tensorContainer instanceof tf.Tensor) {
    valueContainer = await tensorContainer.data();
  } else if (Array.isArray(tensorContainer)) {
    // Start value downloads from all tensors.
    const valuePromiseContainer = tensorContainer.map(async item => {
      if (item instanceof tf.Tensor) {
        return item.data();
      }
      return item;
    });
    // Wait until all values are downloaded.
    valueContainer = await Promise.all(valuePromiseContainer);
  } else if (tensorContainer != null && typeof tensorContainer === 'object') {
    const valuePromiseContainer = [];
    // Start value downloads from all tensors.
    for (const property in tensorContainer) {
      if (tensorContainer[property] instanceof tf.Tensor) {
        valuePromiseContainer.push(tensorContainer[property].data());
      } else {
        valuePromiseContainer.push(tensorContainer[property]);
      }
    }
    // Wait until all values are downloaded.
    valueContainer = await Promise.all(valuePromiseContainer);
  }
  return valueContainer;
}

/**
 * Executes the predict function for `model` (`model.predict` for
 * tf.LayersModel and `model.executeAsync` for tf.GraphModel) and returns a
 * promise that resolves with information about the memory usage:
 * - `newBytes`: the number of new bytes allocated.
 * - `newTensors`: the number of new tensors created.
 * - `peakBytes`: the peak number of bytes allocated.
 * - `kernels`: an array of kernel information objects about their input and
 * output shapes, number of bytes used, number of new tensors created and kernel
 * time (ms). The array is sorted by `kernelTimeMs` field in non-ascending
 * order.
 * - `aggregatedKernels`: an array of aggregated kernel information objects with
 * `name` and `timeMs` fields. The array is sorted by `timeMs` field in
 * non-ascending order.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const profileInfo = await profileModelInference(model, zeros);
 *
 * console.log(`newBytes: ${profileInfo.newBytes}`);
 * console.log(`newTensors: ${profileInfo.newTensors}`);
 * console.log(`peakBytes: ${profileInfo.peakBytes}`);
 * ```
 *
 * @param model An instance of tf.GraphModel or tf.LayersModel for profiling
 *     memory usage in the inference process.
 * @param input The input tensor container for model inference.
 */
async function profileModelInference(model, input) {
  const predict = getPredictFnForModel(model, input);
  return profileInference(predict);
}

/**
 * Executes `predict()` and returns a promise that resolves with information
 * about the memory usage:
 * - `newBytes`: the number of new bytes allocated.
 * - `newTensors`: the number of new tensors created.
 * - `peakBytes`: the peak number of bytes allocated.
 * - `kernels`: an array of kernel information objects about their input and
 * output shapes, number of bytes used, number of new tensors created and kernel
 * time (ms). The array is sorted by `kernelTimeMs` field in non-ascending
 * order.
 * - `aggregatedKernels`: an array of aggregated kernel information objects with
 * `name` and `timeMs` fields. The array is sorted by `timeMs` field in
 * non-ascending order.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const profileInfo = await profileInference(() =>
 * model.predict(zeros));
 *
 * console.log(`newBytes: ${profileInfo.newBytes}`);
 * console.log(`newTensors: ${profileInfo.newTensors}`);
 * console.log(`peakBytes: ${profileInfo.peakBytes}`);
 * ```
 *
 * @param predict The predict function to execute for profiling memory usage.
 */
async function profileInference(predict) {
  if (typeof predict !== 'function') {
    throw new Error(
        'The first parameter should be a function, while ' +
        `a(n) ${typeof predict} is found.`);
  }

  const kernelInfo = await tf.profile(async () => {
    const res = await predict();
    await downloadValuesFromTensorContainer(res);
    tf.dispose(res);
  });

  kernelInfo.kernels =
      kernelInfo.kernels.sort((a, b) => b.kernelTimeMs - a.kernelTimeMs);
  kernelInfo.aggregatedKernels = aggregateKernelTime(kernelInfo.kernels);
  return kernelInfo;
}

/**
 * Aggregate kernels by name and sort the array in non-ascending order of time.
 * Return an array of objects with `name` and `timeMs` fields.
 *
 * @param {Array<Object>} kernels An array of kernel information objects. Each
 *     object must include `name` (string) and `kernelTimeMs` (number) fields.
 */
function aggregateKernelTime(kernels) {
  const aggregatedKernelTime = {};
  kernels.forEach(kernel => {
    const oldAggregatedKernelTime = aggregatedKernelTime[kernel.name];
    if (oldAggregatedKernelTime == null) {
      aggregatedKernelTime[kernel.name] = kernel.kernelTimeMs;
    } else {
      aggregatedKernelTime[kernel.name] =
          oldAggregatedKernelTime + kernel.kernelTimeMs;
    }
  });

  return Object.entries(aggregatedKernelTime)
      .map(([name, timeMs]) => ({name, timeMs}))
      .sort((a, b) => b.timeMs - a.timeMs);
}

/**
 * This map descripes tunable flags and theior corresponding types.
 *
 * The flags (keys) in the map satisfy the following two conditions:
 * - Is tunable. For example, `IS_BROWSER` and `IS_CHROME` is not tunable,
 * because they are fixed when running the scripts.
 * - Does not depend on other flags when registering in `ENV.registerFlag()`.
 * This rule aims to make the list streamlined, and, since there are
 * dependencies between flags, only modifying an independent flag without
 * modifying its dependents may cause inconsistency.
 * (`WEBGL_RENDER_FLOAT32_CAPABLE` is an exception, because only exposing
 * `WEBGL_FORCE_F16_TEXTURES` may confuse users.)
 */
const TUNABLE_FLAG_VALUE_RANGE_MAP = {
  WEBGL_VERSION: [1, 2],
  WASM_HAS_SIMD_SUPPORT: [true, false],
  WASM_HAS_MULTITHREAD_SUPPORT: [true, false],
  WEBGL_CPU_FORWARD: [true, false],
  WEBGL_PACK: [true, false],
  WEBGL_FORCE_F16_TEXTURES: [true, false],
  WEBGL_RENDER_FLOAT32_CAPABLE: [true, false],
  WEBGL_FLUSH_THRESHOLD: [-1, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
};

/**
 * Set environment flags for testing.
 *
 * This is a wrapper function of `tf.env().setFlags()` to constrain users to
 * only set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`).
 *
 * ```js
 * const flagConfig = {
 *        WEBGL_PACK: false,
 *      };
 * await setEnvFlags(flagConfig);
 *
 * console.log(tf.env().getBool('WEBGL_PACK')); // false
 * console.log(tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')); // false
 * ```
 *
 * @param flagConfig An object to store flag-value pairs.
 */
async function setEnvFlags(flagConfig) {
  if (flagConfig == null) {
    return;
  } else if (typeof flagConfig !== 'object') {
    throw new Error(
        `An object is expected, while a(n) ${typeof flagConfig} is found.`);
  }

  // Check the validation of flags and values.
  for (const flag in flagConfig) {
    // TODO: check whether flag can be set as flagConfig[flag].
    if (!(flag in TUNABLE_FLAG_VALUE_RANGE_MAP)) {
      throw new Error(`${flag} is not a tunable or valid environment flag.`);
    }
    if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag].indexOf(flagConfig[flag]) === -1) {
      throw new Error(
          `${flag} value is expected to be in the range [${
              TUNABLE_FLAG_VALUE_RANGE_MAP[flag]}], while ${flagConfig[flag]}` +
          ' is found.');
    }
  }

  tf.env().setFlags(flagConfig);

  // `WASM_HAS_SIMD_SUPPORT` and `WEBGL_VERSION` are also evaluated when
  // initializing backends, not only inferring.
  // TODO: The following backend rebuild logics can be implemented in `setHook`
  // when registering these flags.
  if ('WASM_HAS_SIMD_SUPPORT' in flagConfig) {
    await resetBackend('wasm');
  }

  if ('WEBGL_VERSION' in flagConfig) {
    await resetBackend('webgl');
  }
}

/**
 * Reset the target backend.
 *
 * @param backendName The name of the backend to be reset.
 */
async function resetBackend(backendName) {
  const ENGINE = tf.engine();
  if (!(backendName in ENGINE.registryFactory)) {
    throw new Error(`${backendName} backend is not registed.`);
  }

  const currentBackend = tf.getBackend();

  if (backendName in ENGINE.registry) {
    const backendFactory = tf.findBackendFactory(backendName);
    tf.removeBackend(backendName);
    tf.registerBackend(backendName, backendFactory);
  }

  if (currentBackend === backendName) {
    await tf.setBackend(backendName);
  }
}
