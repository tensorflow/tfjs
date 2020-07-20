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
  wasm: ['WASM_HAS_SIMD_SUPPORT'],
  webgl: [
    'WEBGL_VERSION', 'WEBGL_CPU_FORWARD', 'WEBGL_PACK',
    'WEBGL_FORCE_F16_TEXTURES', 'WEBGL_RENDER_FLOAT32_CAPABLE'
  ]
};
const TUNABLE_FLAG_NAME_MAP = {
  PROD: 'production mode',
  WEBGL_VERSION: 'webgl version',
  WASM_HAS_SIMD_SUPPORT: 'wasm SIMD',
  WEBGL_CPU_FORWARD: 'cpu forward',
  WEBGL_PACK: 'webgl pack',
  WEBGL_FORCE_F16_TEXTURES: 'enforce float16',
  WEBGL_RENDER_FLOAT32_CAPABLE: 'enable float32'
};

/**
 * Records each flag's default value under the runtime environment and is a
 * constant in runtime.
 */
let TUNABLE_FLAG_DEFAULT_VALUE_MAP;

/**
 * Set up flag settings under the UI element of `folderController`:
 * - If it is the first call, show flag settings for both the general and the
 * given backend.
 * - Else, clean up flag settings for the previous backend and show flag
 * settings for the new backend.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
async function showFlagSettings(folderController, backendName) {
  // Clean up flag settings for the previous backend.
  // The first constroller under the `folderController` is the backend setting.
  const fixedSelectionCount = BACKEND_FLAGS_MAP.general.length + 1;
  while (folderController.__controllers.length > fixedSelectionCount) {
    folderController.remove(
        folderController.__controllers[fixedSelectionCount]);
  }

  if (TUNABLE_FLAG_DEFAULT_VALUE_MAP == null) {
    await initDefaultValueMap();
  }

  // Show general flag settings for the first call.
  if (folderController.__controllers.length < fixedSelectionCount) {
    showBackendFlagSettings(folderController, 'general');
  }

  // Show flag settings for the new backend.
  showBackendFlagSettings(folderController, backendName);
}

/**
 * Show flag settings for the given backend under the UI element of
 * `folderController`.
 *
 * @param {dat.gui.GUI} folderController
 * @param {string} backendName
 */
function showBackendFlagSettings(folderController, backendName) {
  const tunableFlags = BACKEND_FLAGS_MAP[backendName];
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
      flagController = folderController.add(state.flags, flag);
    } else {
      // Show dropdown for other types of flags.
      flagController = folderController.add(state.flags, flag, flagValueRange);

      // Because dat.gui always casts dropdown option values to string, we need
      // `stringValueMap` and `onFinishChange()` to recover the value type.
      const stringValueMap = {};
      for (let index = 0; index < flagValueRange.length; index++) {
        const realValue = flagValueRange[index];
        const stringValue = String(flagValueRange[index]);
        stringValueMap[stringValue] = realValue;
      }
      flagController.onFinishChange(stringValue => {
        state.flags[flag] = stringValueMap[stringValue];
      });
    }
    flagController.name(flagName).onChange(() => {
      state.isFlagChanged = true;
    });
  }
}

/**
 * Query all tunable flags' default value and populate `state.flags` with them.
 */
async function initDefaultValueMap() {
  // Clean up the cache to query tunable flags' default values.
  setEnvFlags({});
  TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
  for (const flag in TUNABLE_FLAG_VALUE_RANGE_MAP) {
    TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag] = await tf.env().getAsync(flag);
  }

  // Initialize state.flags with tunable flags' default values.
  for (const flag in TUNABLE_FLAG_DEFAULT_VALUE_MAP) {
    state.flags[flag] = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  }
  state.isFlagChanged = false;
}

/**
 * Heuristically determine flag's value range based on flag's  default value.
 *
 * @param {string} flag
 */
function getTunableRange(flag) {
  const defaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  if (flag === 'WEBGL_FORCE_F16_TEXTURES') {
    return [false, true];
  } else if (typeof defaultValue === 'boolean') {
    return defaultValue ? [false, true] : [false];
  } else if (
      typeof defaultValue === 'number' && defaultValue % 1 === 0 &&
      defaultValue > 1) {
    const tunableRange = [];
    for (let value = 1; value <= defaultValue; value++) {
      tunableRange.push(value);
    }
    return tunableRange;
  } else {
    return [defaultValue];
  }
}
