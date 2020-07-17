const BACKEND_FLAGS_MAP = {
  general: [],
  cpu: [],
  wasm: ['WASM_HAS_SIMD_SUPPORT'],
  webgl: [
    'WEBGL_VERSION', 'WEBGL_CPU_FORWARD', 'WEBGL_PACK',
    'WEBGL_FORCE_F16_TEXTURES', 'WEBGL_RENDER_FLOAT32_CAPABLE'
  ],
};
const TUNABLE_FLAG_NAME_MAP = {
  PROD: 'production mode',
  WEBGL_VERSION: 'webgl version',
  WASM_HAS_SIMD_SUPPORT: 'wasm SIMD',
  WEBGL_CPU_FORWARD: 'cpu forward',
  WEBGL_PACK: 'webgl pack',
  WEBGL_FORCE_F16_TEXTURES: 'enforce float16',
  WEBGL_RENDER_FLOAT32_CAPABLE: 'enable float32',
};

let TUNABLE_FLAG_DEFAULT_VALUE_MAP;
async function showFlagsSettings(folderController) {
  // Delete settings for other flags.
  // The first constroller under the `folderController` is the backend. In
  // addition to backend setting, general flags also apply to all backends.
  const fixedSelectionNum = BACKEND_FLAGS_MAP.general + 1;
  while (folderController.__controllers.length > 1) {
    folderController.remove(folderController.__controllers[1]);
  }

  if (TUNABLE_FLAG_DEFAULT_VALUE_MAP == null) {
    await getFlagDefaultValueMap();
  }

  // Add flag setting for the current backend.
  const tunableFlags = BACKEND_FLAGS_MAP[state.backend];
  for (let index = 0; index < tunableFlags.length; index++) {
    const flag = tunableFlags[index];
    const flagName = TUNABLE_FLAG_NAME_MAP[flag] || flag;
    const flagDefaultValue = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];

    const flagValueRange = getTunableRange(flagDefaultValue);
    // Will not show setting for untunable flags.
    if (flagValueRange.length < 2) {
      continue;
    }

    if (typeof state.flags[flag] === 'boolean') {
      // Show checkbox for boolean flags.
      folderController.add(state.flags, flag).name(flagName);
    } else {
      // Show dropdown for other types of flags.
      folderController.add(state.flags, flag, flagValueRange).name(flagName);
    }
  }
}

async function getFlagDefaultValueMap() {
  // TODO: query default
  await sleep(1);
  TUNABLE_FLAG_DEFAULT_VALUE_MAP = {};
  for (const flag in TUNABLE_FLAG_VALUE_RANGE_MAP) {
    TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag] =
        TUNABLE_FLAG_VALUE_RANGE_MAP[flag][0];
  }

  // Populate tunable flags' default values to state.flags.
  for (const flag in TUNABLE_FLAG_DEFAULT_VALUE_MAP) {
    state.flags[flag] = TUNABLE_FLAG_DEFAULT_VALUE_MAP[flag];
  }
}

function getTunableRange(defaultValue) {
  if (typeof defaultValue === 'boolean') {
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
