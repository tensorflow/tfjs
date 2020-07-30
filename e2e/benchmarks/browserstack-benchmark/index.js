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

const BROWSER_FIELDS =
    ['os', 'os_version', 'browser', 'browser_version', 'device'];
const socket = io();
const state = {
  run: () => {
    // Disable the benchmark button.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = .8;

    if (state.browser.os === 'ios' || state.browser.os === 'android') {
      state.browser.real_mobile = true;
    }
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
    device: 'null',
    real_mobile: null
  }
};

let browserTreeRoot;
let gui;
let browserFolder;
let benchmarkButton;
let browserSettingControllers = [];

function constructBrowserTree(browsers) {
  const browserTreeRoot = {};
  browsers.forEach(browser => {
    let currentNode = browserTreeRoot;
    for (const field of BROWSER_FIELDS) {
      if (currentNode[browser[field]] == null) {
        currentNode[browser[field]] = {};
      }
      currentNode = currentNode[browser[field]];
    }
    if (currentNode['browserList'] == null) {
      currentNode['browserList'] = [];
    }
    currentNode['browserList'].push(browser);
  });
  return browserTreeRoot;
}

function cleanFollowingBrowserFields(currentFieldController) {
  while (browserFolder.__controllers.length > 0) {
    let tailController =
        browserFolder.__controllers[browserFolder.__controllers.length - 1];
    if (tailController === currentFieldController) {
      break;
    } else {
      browserFolder.remove(tailController);
    }
  }
}

function updateFollowingFields(
    currentFieldIndex, currentFieldValue, currentNode) {
  const nextFieldIndex = currentFieldIndex + 1;
  if (nextFieldIndex === BROWSER_FIELDS.length) {
    return;
  }

  const nextFieldName = BROWSER_FIELDS[nextFieldIndex];
  const nextNode = currentNode[currentFieldValue];
  const nextFieldAvailableValues = Object.keys(nextNode);
  let nextFieldValue = state.browser[nextFieldName];

  // Update the options for the next field.
  const nextFieldController = browserSettingControllers[nextFieldIndex].options(
      nextFieldAvailableValues);
  nextFieldController.onFinishChange(() => {
    const newValue = state.browser[nextFieldName];
    updateFollowingFields(nextFieldIndex, newValue, nextNode);
  });
  browserSettingControllers[nextFieldIndex] = nextFieldController;

  // Update the value of the next field, if the old value is not applicable.
  if (nextNode[nextFieldValue] == null) {
    nextFieldValue = nextFieldAvailableValues[0];
  }
  nextFieldController.setValue(nextFieldValue);

  if (nextFieldValue === 'null') {
    nextFieldController.__li.hidden = true;
  } else {
    nextFieldController.__li.hidden = false;
  }

  updateFollowingFields(nextFieldIndex, nextFieldValue, nextNode);
}

function showBrowserField(field, index, currentNode) {
  const fieldController =
      browserFolder.add(state.browser, field, Object.keys(currentNode));

  fieldController.onFinishChange(() => {
    const newValue = state.browser[field];
    updateFollowingFields(index, newValue, currentNode);
  });
  if (state.browser[field] == 'null') {
    fieldController.__li.hidden = true;
  }
  browserSettingControllers.push(fieldController);
}

function onPageLoad() {
  // Once the server is connected, the server will send back all
  // BrowserStack's available browsers.
  socket.on('availableBrowsers', availableBrowsersArray => {
    if (browserTreeRoot == null) {
      browserTreeRoot = constructBrowserTree(availableBrowsersArray);

      // Show browser settings.
      let currentNode = browserTreeRoot;
      BROWSER_FIELDS.forEach((field, index) => {
        showBrowserField(field, index, currentNode);
        currentNode = currentNode[state.browser[field]];
      });
      browserFolder.open();

      benchmarkButton = gui.add(state, 'run').name('Run benchmark');
    }
  });

  socket.on('benchmarkComplete', benchmarkResult => {
    const {timeInfo, memoryInfo} = benchmarkResult;

    // TODO: Add UI for results presenting.
    document.getElementById('results').innerHTML +=
        JSON.stringify(timeInfo, null, 2);

    // Enable the benchmark button.
    benchmarkButton.__li.style.pointerEvents = '';
    benchmarkButton.__li.style.opacity = 1;
  });

  gui = new dat.gui.GUI({width: 400});
  browserFolder = gui.addFolder('Browser');
}

onPageLoad();
