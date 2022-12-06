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
};
if (tf.engine().backendNames().includes('webgpu')) {
  BACKEND_FLAGS_MAP['webgpu'] =
    ['WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE', 'KEEP_INTERMEDIATE_TENSORS'];
}

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
};
if (tf.engine().backendNames().includes('webgpu')) {
  TUNABLE_FLAG_NAME_MAP['WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE'] =
    'deferred submit batch size';
}

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
    while (folderController.__controllers.length > fixedSelectionCount) {
      folderController.remove(
        folderController
          .__controllers[folderController.__controllers.length - 1]);
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
 * Heuristically determine flag's value range based on flag's default value.
 *
 * Assume that the flag's default value has already chosen the best option for
 * the runtime environment, so users can only tune the flag value downwards.
 *
 * For example, if the default value of `WEBGL_RENDER_FLOAT32_CAPABLE` is false,
 * the tunable range is [false]; otherwise, the tunable range is [true. false].
 *
 * @param {string} flag
 */
function getTunableRange(flag) {
  const defaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  if (flag === 'WEBGL_FORCE_F16_TEXTURES' ||
    flag === 'WEBGL_PACK_DEPTHWISECONV' || 'KEEP_INTERMEDIATE_TENSORS') {
    return [false, true];
  } else if (flag === 'WEBGL_VERSION') {
    const tunableRange = [];
    for (let value = 1; value <= defaultValue; value++) {
      tunableRange.push(value);
    }
    return tunableRange;
  } else if (flag === 'WEBGL_FLUSH_THRESHOLD') {
    const tunableRange = [-1];
    for (let value = 0; value <= 2; value += 0.25) {
      tunableRange.push(value);
    }
    return tunableRange;
  } else if (typeof defaultValue === 'boolean') {
    return defaultValue || flag === 'WEBGL_USE_SHAPES_UNIFORMS' ?
      [false, true] :
      [false];
  } else if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag] != null) {
    return TUNABLE_FLAG_VALUE_RANGE_MAP[flag];
  } else {
    return [defaultValue];
  }
}


async function benchmarkAll() {
  const numRuns = 50;
  const SHARED_DIM = 256;
  const OUTPUT_HEIGHT = 256;
  const OUTPUT_WIDTH = 256;

  tf.env().set('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', true);

  console.log('Varying SHARED_DIM:');
  let resData = [];
  for (let index = 1; index <= 384; index += 1) {
    const variable = index * 128;
    const res = await benchmark(variable, OUTPUT_HEIGHT, OUTPUT_WIDTH, numRuns);
    resData.push(`${variable}\t${res}`);
  }
  console.log(resData.join('\n'));

  console.log('Varying OUTPUT_HEIGHT:');
  resData = [];
  for (let index = 1; index <= 384; index += 1) {
    const variable = index * 128;
    const res = await benchmark(SHARED_DIM, variable, OUTPUT_WIDTH, numRuns);
    resData.push(`${variable}\t${res}`);
  }
  console.log(resData.join('\n'));

  // Vary width
  console.log('Varying OUTPUT_WIDTH:');
  resData = [];
  for (let index = 1; index <= 384; index += 1) {
    const variable = index * 128;
    const res = await benchmark(SHARED_DIM, OUTPUT_HEIGHT, variable, numRuns);
    resData.push(`${variable}\t${res}`);
  }
  console.log(resData.join('\n'));
}

async function getPerf(sharedD, height, width, numRuns) {
  const a = tf.ones([height, sharedD]);
  const b = tf.ones([sharedD, width]);
  const inf = () => tf.matMul(a, b);
  await tf.profile(inf);

  const res = [];
  for (let i = 0; i < numRuns; i++) {
    const perf = await tf.profile(inf);
    assert(perf.kernels[0].extraInfo.match(/:/g).length === 1, perf.kernels[0].extraInfo.match(/:/g));
    assert(perf.kernels[0].extraInfo.includes('MatMulPackedProgram'), 'Is not running MRT matmul');
    res.push(perf.kernels[0]);
  }
  return res;
}

async function benchmark(sharedD, height, width, numRuns) {
  console.log('running setting: ', [sharedD, height, width]);
  const res = (await getPerf(sharedD, height, width, numRuns)).map(e => e.kernelTimeMs);
  return res.reduce((prev, cur) => prev + cur, 0) / numRuns;
}

function size(shape) {
  shape.reduce(
    (previousValue, currentValue) => previousValue * currentValue, 1
  );
}

function assert(bool, msg = 'assert failed!') {
  if (!bool) {
    throw new Error(msg);
  }
}
