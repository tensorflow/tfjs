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

const BACKEND_FLAGS_MAP = {
  general: [],
  cpu: [],
  wasm: [
    'WASM_HAS_SIMD_SUPPORT',
    'WASM_HAS_MULTITHREAD_SUPPORT',
    'CHECK_COMPUTATION_FOR_ERRORS',
    'KEEP_INTERMEDIATE_TENSORS',
  ],
  webgl: [
    'WEBGL_VERSION', 'WEBGL_CPU_FORWARD', 'WEBGL_PACK',
    'WEBGL_FORCE_F16_TEXTURES', 'WEBGL_RENDER_FLOAT32_CAPABLE',
    'WEBGL_FLUSH_THRESHOLD', 'WEBGL_PACK_DEPTHWISECONV',
    'CHECK_COMPUTATION_FOR_ERRORS', 'WEBGL_USE_SHAPES_UNIFORMS',
    'KEEP_INTERMEDIATE_TENSORS'
  ],
  tflite: [],
  webgpu: ['WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 'KEEP_INTERMEDIATE_TENSORS']
};

const TUNABLE_FLAG_NAME_MAP = {
  PROD: 'production mode',
  WEBGL_VERSION: 'webgl version',
  WASM_HAS_SIMD_SUPPORT: 'wasm SIMD',
  WASM_HAS_MULTITHREAD_SUPPORT: 'wasm multithread',
  WEBGL_CPU_FORWARD: 'cpu forward',
  WEBGL_PACK: 'webgl pack',
  WEBGL_FORCE_F16_TEXTURES: 'enforce float16',
  WEBGL_RENDER_FLOAT32_CAPABLE: 'enable float32',
  WEBGL_FLUSH_THRESHOLD: 'GL flush wait time(ms)',
  WEBGL_PACK_DEPTHWISECONV: 'Packed depthwise Conv2d',
  WEBGL_USE_SHAPES_UNIFORMS: 'Use shapes uniforms',
  CHECK_COMPUTATION_FOR_ERRORS: 'Check each op result',
  KEEP_INTERMEDIATE_TENSORS: 'Print intermediate tensors',
  WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE: 'deferred submit batch size',
};

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
  WEBGL_FLUSH_THRESHOLD: [-1, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
  WEBGL_PACK_DEPTHWISECONV: [true, false],
  CHECK_COMPUTATION_FOR_ERRORS: [true, false],
  KEEP_INTERMEDIATE_TENSORS: [true, false],
  WEBGL_USE_SHAPES_UNIFORMS: [true, false],
  WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE: [1, 5, 10, 15, 20, 25, 30, 35, 40]
};

/**
 * Records each flag's default value under the runtime environment and is a
 * constant in runtime.
 */
let TUNABLE_FLAG_DEFAULT_VALUE_MAP;

/**
 * Set up flag settings under the UI element of `folderController`:
 * - If it is the first call, initialize the flags' default value and show flag
 * settings for both the general and the given backend.
 * - Else, clean up flag settings for the previous backend and show flag
 * settings for the new backend.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
async function showFlagSettingsAndReturnTunableFlagControllers(
    folderController, backendName) {
  // Determine wether it is the first call.
  if (TUNABLE_FLAG_DEFAULT_VALUE_MAP == null) {
    await initDefaultValueMap();
    showBackendFlagSettingsAndReturnTunableFlagControllers(
        folderController, 'general');
  } else {
    // Clean up flag settings for the previous backend.
    // The first constroller under the `folderController` is the backend
    // setting.
    const fixedSelectionCount = BACKEND_FLAGS_MAP.general.length + 1;
    while (folderController.controllers.length > fixedSelectionCount) {
      folderController.controllers[folderController.controllers.length - 1]
          .destroy();
    }
  }

  // Show flag settings for the new backend and return the tunable flags
  // controllers.
  return showBackendFlagSettingsAndReturnTunableFlagControllers(
      folderController, backendName);
}

const stringValueMap = {};
/**
 * Show flag settings for the given backend under the UI element of
 * `folderController`.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
function showBackendFlagSettingsAndReturnTunableFlagControllers(
    folderController, backendName) {
  const tunableFlags = BACKEND_FLAGS_MAP[backendName];
  const tunableFlagControllers = {};

  // Remove it once we figure out how to correctly read the tensor data
  // before the tensor is disposed in profiling mode.
  if (backendName === 'webgpu' &&
      state.flags['CHECK_COMPUTATION_FOR_ERRORS'] === true) {
    state.flags['CHECK_COMPUTATION_FOR_ERRORS'] = false;
    state.isFlagChanged = true;
  }

  for (let index = 0; index < tunableFlags.length; index++) {
    const flag = tunableFlags[index];
    const flagName = TUNABLE_FLAG_NAME_MAP[flag] || flag;

    // When tunable (bool) and range (array) attributes of `flagRegistry` is
    // implemented, we can apply them to here.
    const flagValueRange = getTunableRange(flag);
    // Heuristically consider a flag with at least two options as tunable.
    if (flagValueRange.length < 2) {
      console.warn(
          `The ${flag} is considered as untunable, ` +
          `because its value range is [${flagValueRange}].`);
      continue;
    }
    let flagController;
    if (typeof flagValueRange[0] === 'boolean') {
      // Show checkbox for boolean flags.
      try {
        flagController = folderController.add(state.flags, flag);
      } catch (ex) {
        console.warn(ex.message);
        continue;
      }
    } else {
      // Show dropdown for other types of flags.
      try {
        flagController =
            folderController.add(state.flags, flag, flagValueRange);
      } catch (ex) {
        console.warn(ex.message);
        continue;
      }
      // Because dat.gui always casts dropdown option values to string, we need
      // `stringValueMap` and `onFinishChange()` to recover the value type.
      if (stringValueMap[flag] == null) {
        stringValueMap[flag] = {};
        for (let index = 0; index < flagValueRange.length; index++) {
          const realValue = flagValueRange[index];
          const stringValue = String(flagValueRange[index]);
          stringValueMap[flag][stringValue] = realValue;
        }
      }
      flagController.onFinishChange(stringValue => {
        state.flags[flag] = stringValueMap[flag][stringValue];
      });
    }
    flagController.name(flagName).onChange(() => {
      state.isFlagChanged = true;
    });
    tunableFlagControllers[flag] = flagController;
  }
  return tunableFlagControllers;
}

/**
 * Query all tunable flags' default value and populate `state.flags` with them.
 */
async function initDefaultValueMap() {
  // Clean up the cache to query tunable flags' default values.
  setEnvFlags({});
  TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
  for (const backend in BACKEND_FLAGS_MAP) {
    for (let index = 0; index < BACKEND_FLAGS_MAP[backend].length; index++) {
      const flag = BACKEND_FLAGS_MAP[backend][index];
      try {
        TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag] = await tf.env().getAsync(flag);
      } catch (ex) {
        console.warn(ex.message);
      }
    }
  }

  // Initialize state.flags with tunable flags' default values.
  for (const flag in TUNABLE_FLAG_DEFAULT_VALUE_MAP) {
    state.flags[flag] = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  }
  state.isFlagChanged = false;
}

/**
 * Determine flag's value range.
 *
 * @param {string} flag
 */
function getTunableRange(flag) {
  const defaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag] != null) {
    return TUNABLE_FLAG_VALUE_RANGE_MAP[flag];
  } else {
    return [defaultValue];
  }
}

/**
 * Set environment flags for testing.
 *
 * This will first set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`). Then
 * set URL parameter flags. If there are overlap, URL parameter flags will
 * override tunable flags.
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
    return true;
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
  setEnvFlagsFromUrlParameters();

  // `WASM_HAS_SIMD_SUPPORT` and `WEBGL_VERSION` are also evaluated when
  // initializing backends, not only inferring.
  // TODO: The following backend rebuild logics can be implemented in `setHook`
  // when registering these flags.
  if ('WASM_HAS_SIMD_SUPPORT' in flagConfig) {
    return await resetBackend('wasm');
  }

  if ('WEBGL_VERSION' in flagConfig) {
    return await resetBackend('webgl');
  }
}

/**
 * Set flags from URL. URL should be in the format:
 * ?tfjsflags=FLAG1:1,FLAG2:true.
 */
function setEnvFlagsFromUrlParameters() {
  const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
  const urlParams = tf.env().getQueryParams(location.search);
  if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
    const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
    keyValues.forEach(keyValue => {
      const [key, value] = keyValue.split(':');
      try {
        tf.env().set(key, parseValue(value));
      } catch (err) {
        console.error(err);
      }
    });
  }
}

/**
 * Converted a URL parameter to a typed value, such a boolean, number, string.
 */
function parseValue(value) {
  const lowerCaseValue = value.toLowerCase();
  if (lowerCaseValue === 'true' || lowerCaseValue === 'false') {
    return lowerCaseValue === 'true';
  } else if (`${+ lowerCaseValue}` === lowerCaseValue) {
    return +lowerCaseValue;
  } else {
    return value;
  }
}
