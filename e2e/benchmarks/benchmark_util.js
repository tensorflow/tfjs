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
 * Executes the predict function for `model` and times the inference process for
 * `numRuns` rounds. Then returns a promise that resolves with an array of
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
 * @param numRuns The number of rounds for timing the inference process.
 */
async function profileInferenceTimeForModel(model, input, numRuns = 1) {
  let predict;
  if (model instanceof tf.GraphModel) {
    predict = () => model.executeAsync(input);
  } else if (model instanceof tf.LayersModel) {
    predict = () => model.predict(input);
  } else {
    throw new Error(
        'Please pass in an instance of tf.GraphModel ' +
        'or tf.LayersModel as the first parameter.');
  }
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
 * Asynchronously downloads the values in parallel from any `tf.Tensor`s found
 * within the provided object. Returns a promise of `TypedArray` or
 * `TypedArray[]` that resolves when the computation has finished.
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

const TUNABLE_FLAGS = {
  WEBGL_VERSION: 'number',
  WASM_HAS_SIMD_SUPPORT: 'boolean',
  WEBGL_CPU_FORWARD: 'boolean',
  WEBGL_PACK: 'boolean',
  WEBGL_FORCE_F16_TEXTURES: 'boolean',
  WEBGL_RENDER_FLOAT32_CAPABLE: 'boolean',
};

/**
 * Set environment flags in `TUNABLE_FLAGS` list.
 *
 * This is a wrapper function of `tf.env().setFlags()`, and this function adds
 * flag checking and re-construct environment variables.
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
    if (!(flag in TUNABLE_FLAGS)) {
      throw new Error(`${flag} is not a tunable or valid environment flag.`);
    }
    if (typeof flagConfig[flag] !== TUNABLE_FLAGS[flag]) {
      throw new Error(
          `${flag} is expected to be a ${TUNABLE_FLAGS[flag]}, while a(n) ` +
          `${typeof flagConfig[flag]} is found.`);
    }
  }

  tf.env().setFlags(flagConfig);

  // `WASM_HAS_SIMD_SUPPORT` and `WEBGL_VERSION` are also evaluated when
  // initializing backends, not only inferring.
  // TODO: The following backend rebuild logics can be implemented in `setHook`
  // when registering these flags.
  const ENGINE = tf.engine();
  if ('WASM_HAS_SIMD_SUPPORT' in flagConfig && 'wasm' in ENGINE.registry) {
    const currentBackend = tf.getBackend();
    const wasmFactory = tf.findBackendFactory('wasm');
    tf.removeBackend('wasm');
    tf.registerBackend('wasm', wasmFactory);

    if (currentBackend === 'wasm') {
      await tf.setBackend('wasm');
    }
  }

  if ('WEBGL_VERSION' in flagConfig && 'webgl' in ENGINE.registry) {
    const currentBackend = tf.getBackend();
    const webglFactory = tf.findBackendFactory('webgl');
    tf.removeBackend('webgl');
    tf.registerBackend('webgl', webglFactory);

    if (currentBackend === 'webgl') {
      await tf.setBackend('webgl');
    }
  }
}
