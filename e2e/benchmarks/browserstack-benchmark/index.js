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

const TUNABLE_BROWSER_FIELDS =
    ['os', 'os_version', 'browser', 'browser_version', 'device'];
const state = {
  run: () => {
    // Disable the benchmark button, until the benchmark is complete.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = .8;

    // Send the benchmark configuration to the server to start the benchmark.
    if (state.browser.device === 'null') {
      state.browser.device = null;
    }
    socket.emit('run', {browsers: [state.browser]});
  },
  browser: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: '84.0',
    os: 'OS X',
    os_version: 'Catalina',
    device: 'null'
  }
};

let socket;

// UI controllers.
let gui;
let browserFolder;
let benchmarkButton;
let browserSettingControllers = [];

// Available BrowserStack's browsers will be collected in a tree when the array
// of available browsers is recieved in runtime.
let browserTreeRoot;

/**
 * Collect all given browsers to a tree. The field of each level is defined by
 * `TUNABLE_BROWSER_FIELDS`.
 *
 * The tree node is implemented by Map/Object:
 * - For non-leaf nodes, each node stores a map: each key is the index to a
 * child node and the correspoding value is the child node.
 * - For leaf nodes, each leaf node stores the full configuration for a certain
 * browser.
 *
 * @param {Array<object>} browsersArray An array of browser configurations.
 */
function constructBrowserTree(browsersArray) {
  const browserTreeRoot = {};
  browsersArray.forEach(browser => {
    let currentNode = browserTreeRoot;

    // Route through non-leaf nodes.
    for (let fieldIndex = 0; fieldIndex <= TUNABLE_BROWSER_FIELDS.length - 2;
         fieldIndex++) {
      const fieldName = TUNABLE_BROWSER_FIELDS[fieldIndex];
      if (currentNode[browser[fieldName]] == null) {
        currentNode[browser[fieldName]] = {};
      }
      currentNode = currentNode[browser[fieldName]];
    }

    // Set the full configuration as the leaf node.
    const leafFieldName =
        TUNABLE_BROWSER_FIELDS[TUNABLE_BROWSER_FIELDS.length - 1];
    const leafFieldValue = browser[leafFieldName];
    if (currentNode[leafFieldValue] == null) {
      currentNode[leafFieldValue] = browser;
    } else {
      console.warn(
          `The browser ${browser} shares the same ` +
          'configuration with another browser.');
    }
  });
  return browserTreeRoot;
}

/**
 * Once the value of a certain browser field is changed, the values and options
 * of the following fields will be invalid. This function updates the following
 * fields recursively and does nothing for the leaf nodes.
 *
 * @param {number} currentFieldIndex
 * @param {string} currentFieldValue
 * @param {object} currentNode
 */
function updateFollowingFields(
    currentFieldIndex, currentFieldValue, currentNode) {
  const nextFieldIndex = currentFieldIndex + 1;
  if (nextFieldIndex === TUNABLE_BROWSER_FIELDS.length) {
    return;
  }

  const nextFieldName = TUNABLE_BROWSER_FIELDS[nextFieldIndex];
  const nextNode = currentNode[currentFieldValue];
  const nextFieldAvailableValues = Object.keys(nextNode);
  let nextFieldValue = state.browser[nextFieldName];

  // Update the value of the next field, if the old value is not applicable.
  if (nextNode[nextFieldValue] == null) {
    nextFieldValue = nextFieldAvailableValues[0];
  }

  // Update the options for the next field.
  const nextFieldController = browserSettingControllers[nextFieldIndex].options(
      nextFieldAvailableValues);

  // When updating options for a dat.gui controller, a new controller instacne
  // will be created, so we need to bind the event again and record the new
  // controller.
  nextFieldController.onFinishChange(() => {
    const newValue = state.browser[nextFieldName];
    updateFollowingFields(nextFieldIndex, newValue, nextNode);
  });
  browserSettingControllers[nextFieldIndex] = nextFieldController;

  nextFieldController.setValue(nextFieldValue);

  if (nextFieldValue === 'null') {
    nextFieldController.__li.hidden = true;
  } else {
    nextFieldController.__li.hidden = false;
  }

  updateFollowingFields(nextFieldIndex, nextFieldValue, nextNode);
}

/**
 * This is a wrapper function of `dat.gui.GUI.add()` with:
 * - Binds the `finishChange` event to update the value and options for the
 * controller of its child field.
 * - Hides the dropdown menu, if the field is not applicable for this browser.
 *
 * @param {number} fieldIndex The index of the browser field to be shown.
 * @param {object} currentNode The keys of this map are available values
 *     for this field.
 */
function showBrowserField(fieldIndex, currentNode) {
  const fieldName = TUNABLE_BROWSER_FIELDS[fieldIndex];
  const fieldController =
      browserFolder.add(state.browser, fieldName, Object.keys(currentNode));

  fieldController.onFinishChange(() => {
    const newValue = state.browser[fieldName];
    updateFollowingFields(fieldIndex, newValue, currentNode);
  });

  // Null represents the field is not applicable for this browser. For example,
  // `browser_version` is normally not applicable for mobile devices.
  if (state.browser[fieldName] === 'null') {
    fieldController.__li.hidden = true;
  }

  // The controller will be used to reset options when the value of its parent
  // field is changed.
  browserSettingControllers.push(fieldController);
}

function onPageLoad() {
  socket = io();

  // Once the server is connected, the server will send back all
  // BrowserStack's available browsers in an array.
  socket.on('availableBrowsers', availableBrowsersArray => {
    if (browserTreeRoot == null) {
      // Initialize the browser tree.
      browserTreeRoot = constructBrowserTree(availableBrowsersArray);

      // Show browser settings.
      let currentNode = browserTreeRoot;
      TUNABLE_BROWSER_FIELDS.forEach((field, index) => {
        showBrowserField(index, currentNode);
        currentNode = currentNode[state.browser[field]];
      });
      browserFolder.open();

      // Enable users to benchmark.
      benchmarkButton = gui.add(state, 'run').name('Run benchmark');
    }
  });

  socket.on('benchmarkComplete', benchmarkResult => {
    const {timeInfo, memoryInfo} = benchmarkResult;

    // TODO: Add UI to present results.
    document.getElementById('results').innerHTML +=
        JSON.stringify(timeInfo, null, 2);

    // Enable users to benchmark again.
    benchmarkButton.__li.style.pointerEvents = '';
    benchmarkButton.__li.style.opacity = 1;
  });

  gui = new dat.gui.GUI({width: 400});
  browserFolder = gui.addFolder('Browser');
}
