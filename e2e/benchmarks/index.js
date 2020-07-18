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

// This map depends on the runtime environment.
let TUNABLE_FLAG_DEFAULT_VALUE_MAP;

// Set up flag settings from scratch or for a new backend.
async function showFlagSettings(folderController, backendName) {
  // Delete settings for other flags.
  // The first constroller under the `folderController` is the backend.
  // Backend setting and general flag setting apply to all backends.
  const fixedSelectionNum = BACKEND_FLAGS_MAP.general.length + 1;
  while (folderController.__controllers.length > fixedSelectionNum) {
    folderController.remove(folderController.__controllers[fixedSelectionNum]);
  }

  if (TUNABLE_FLAG_DEFAULT_VALUE_MAP == null) {
    await initDefaultValueMap();
  }

  // Add general flag settings.
  if (folderController.__controllers.length < fixedSelectionNum) {
    showBackendFlagSettings(folderController, 'general');
  }
  // Add flag setting for the new backend.
  showBackendFlagSettings(folderController, backendName);
}

function showBackendFlagSettings(folderController, backendName) {
  const tunableFlags = BACKEND_FLAGS_MAP[backendName];
  for (let index = 0; index < tunableFlags.length; index++) {
    const flag = tunableFlags[index];
    const flagName = TUNABLE_FLAG_NAME_MAP[flag] || flag;

    // When tunable (bool) and range (array) attributes of `flagRegistry` is
    // implemented, we can apply them to here.
    const flagValueRange = getTunableRange(flag);
    // Consider a flag with at least two options as tunable.
    if (flagValueRange.length < 2) {
      continue;
    }

    let flagController;
    if (typeof flagValueRange[0] === 'boolean') {
      // Show checkbox for boolean flags.
      flagController = folderController.add(state.flags, flag);
    } else {
      // Show dropdown for other types of flags.
      // Because dat.gui always casts dropdown option values to string, we need
      // `stringValueMap` and `onFinishChange()` to recover the value type.
      const stringValueMap = {};
      flagValueRange.forEach(option => {
        stringValueMap[option] = option;
      });
      flagController = folderController.add(state.flags, flag, flagValueRange)
                           .onFinishChange(stringValue => {
                             state.flags[flag] = stringValueMap[stringValue];
                           });
    }
    flagController.name(flagName).onChange(() => {
      state.isFlagChanged = true;
    });
  }
}

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

// Heuristically determine flag's value range based on the default value.
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
