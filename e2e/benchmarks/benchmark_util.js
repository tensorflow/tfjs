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

  const tensorArray = [];
  try {
    model.inputs.forEach((inputNode, inputNodeIndex) => {
      // Replace -1 or null in input tensor shape.
      const inputShape = inputNode.shape.map(shapeValue => {
        if (shapeValue == null || shapeValue < 0) {
          return 1;
        } else {
          return shapeValue;
        }
      });

      // Construct the input tensor.
      let inputTensor;
      if (inputNode.dtype === 'float32' || inputNode.dtype === 'int32') {
        inputTensor = tf.randomNormal(inputShape, 0, 1000, inputNode.dtype);
      } else {
        throw new Error(
            `The ${inputNode.dtype} dtype of '${inputNode.name}' input ` +
            `at model.inputs[${inputNodeIndex}] is not supported.`);
      }
      tensorArray.push(inputTensor);
    });

    // Return tensor map for tf.GraphModel.
    if (model instanceof tf.GraphModel) {
      const tensorMap = model.inputNodes.reduce((map, inputName, i) => {
        map[inputName] = tensorArray[i];
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
 * for `numRuns` rounds. Then returns a promise that resolves with an array of
 * inference times for each inference process.
 *
 * The inference time contains the time spent by both `predict()` and `data()`
 * called by tensors in the prediction.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const elapsedTimeArray =
 *    await profileInferenceTimeForModel(model, zeros, 2);
 *
 * console.log(`Elapsed time array: ${elapsedTimeArray}`);
 * ```
 *
 * @param model An instance of tf.GraphModel or tf.LayersModel for timing the
 *     inference process.
 * @param input The input tensor container for model inference.
 * @param numRuns The number of rounds for timing the inference process.
 */
async function profileInferenceTimeForModel(model, input, numRuns = 1) {
  const predict = getPredictFnForModel(model, input);
  return profileInferenceTime(predict, numRuns);
}

/**
 * Executes `predict()` and times the inference process for `numRuns` rounds.
 * Then returns a promise that resolves with an array of inference time for each
 * inference process.
 *
 * The inference time contains the time spent by both `predict()` and `data()`
 * called by tensors in the prediction.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const elapsedTimeArray =
 *    await profileInferenceTime(() => model.predict(zeros), 2);
 *
 * console.log(`Elapsed time array: ${elapsedTimeArray}`);
 * ```
 *
 * @param predict The predict function to execute and time.
 * @param numRuns The number of rounds for `predict` to execute and time.
 */
async function profileInferenceTime(predict, numRuns = 1) {
  if (typeof predict !== 'function') {
    throw new Error(
        'The first parameter should be a function, while ' +
        `a(n) ${typeof predict} is found.`);
  }

  const elapsedTimeArray = [];
  for (let i = 0; i < numRuns; i++) {
    const start = performance.now();
    const res = await predict();
    // The prediction can be tf.Tensor|tf.Tensor[]|{[name: string]: tf.Tensor}.
    const value = await downloadValuesFromTensorContainer(res);
    const elapsedTime = performance.now() - start;

    tf.dispose(res);
    elapsedTimeArray.push(elapsedTime);
  }
  return elapsedTimeArray;
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
 * - `newBytes`: the number of new bytes allocated
 * - `newTensors`: the number of new tensors created
 * - `peakBytes`: the peak number of bytes allocated
 * - `kernels`: an array of objects for each kernel involved that reports
 * their input and output shapes, number of bytes used, and number of new
 * tensors created.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const memoryInfo = await profileInferenceMemoryForModel(model, zeros);
 *
 * console.log(`newBytes: ${memoryInfo.newBytes}`);
 * console.log(`newTensors: ${memoryInfo.newTensors}`);
 * console.log(`peakBytes: ${memoryInfo.peakBytes}`);
 * ```
 *
 * @param model An instance of tf.GraphModel or tf.LayersModel for profiling
 *     memory usage in the inference process.
 * @param input The input tensor container for model inference.
 */
async function profileInferenceMemoryForModel(model, input) {
  const predict = getPredictFnForModel(model, input);
  return profileInferenceMemory(predict);
}

/**
 * Executes `predict()` and returns a promise that resolves with information
 * about the memory usage:
 * - `newBytes`: the number of new bytes allocated
 * - `newTensors`: the number of new tensors created
 * - `peakBytes`: the peak number of bytes allocated
 * - `kernels`: an array of objects for each kernel involved that reports
 * their input and output shapes, number of bytes used, and number of new
 * tensors created.
 *
 * ```js
 * const modelUrl =
 *    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
 * const model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});
 * const zeros = tf.zeros([1, 224, 224, 3]);
 * const memoryInfo = await profileInferenceMemory(() =>
 * model.predict(zeros));
 *
 * console.log(`newBytes: ${memoryInfo.newBytes}`);
 * console.log(`newTensors: ${memoryInfo.newTensors}`);
 * console.log(`peakBytes: ${memoryInfo.peakBytes}`);
 * ```
 *
 * @param predict The predict function to execute for profiling memory usage.
 */
async function profileInferenceMemory(predict) {
  if (typeof predict !== 'function') {
    throw new Error(
        'The first parameter should be a function, while ' +
        `a(n) ${typeof predict} is found.`);
  }

  const memoryInfo = await profile(async () => {
    const res = await predict();
    await downloadValuesFromTensorContainer(res);
    tf.dispose(res);
  });
  return memoryInfo;
}

/**
 * This function is temporarily used and will be deleted after a new release of
 * tf-core. This function modifies
 * This function is temporarily used and will be deleted after a new release
 * of tf-core. This function modifies
 * [`tf.profile`](https://github.com/tensorflow/tfjs/blob/95b5f878218ee45c0f8464386ee01d1f96e78297/tfjs-core/src/engine.ts#L848)
 * in the following points:
 * - replaces all `this` by `tf.engine()`
 * - adds `await` in `this.state.activeProfile.result = query();`
 *
 * When deleting this method, please change the caller
 * `profileInferenceMemory`.
 */
async function profile(query) {
  const engine = tf.engine();
  engine.state.profiling = true;

  const startBytes = engine.state.numBytes;
  const startNumTensors = engine.state.numTensors;

  engine.state.activeProfile.kernels = [];
  engine.state.activeProfile.result = await query();

  engine.state.profiling = false;

  engine.state.activeProfile.peakBytes = Math.max(
      ...engine.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
  engine.state.activeProfile.newBytes = engine.state.numBytes - startBytes;
  engine.state.activeProfile.newTensors =
      engine.state.numTensors - startNumTensors;
  return engine.state.activeProfile;
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
  WEBGL_CPU_FORWARD: [true, false],
  WEBGL_PACK: [true, false],
  WEBGL_FORCE_F16_TEXTURES: [true, false],
  WEBGL_RENDER_FLOAT32_CAPABLE: [true, false],
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
