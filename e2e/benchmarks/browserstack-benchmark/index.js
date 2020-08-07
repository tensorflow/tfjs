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

const socket = io();
const state = {
  run: () => {
    // Disable the button.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = .5;

    const tabName = createTab(state.browser);

    // Send the benchmark configuration to the server to start the benchmark.
    if (state.browser.device === 'null') {
      state.browser.device = null;
    }
    const benchmark = {...state.benchmark};
    if (state.benchmark.model !== 'custom') {
      delete benchmark['modelUrl'];
    }

    socket.emit('run', {tabName, benchmark, browser: state.browser});
  },
  browser: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: '84.0',
    os: 'OS X',
    os_version: 'Catalina',
    device: 'null'
  },
  benchmark: {model: 'mobilenet_v2', modelUrl: '', numRuns: 1, backend: 'wasm'}
};

const nameCounter = {};
function getTabName(browserConf) {
  let baseName;
  if (browserConf.os === 'android' || browserConf.os === 'ios') {
    baseName = browserConf.device;
  } else {
    baseName = `${browserConf.os}(${browserConf.os_version})`;
  }
  if (nameCounter[baseName] == null) {
    nameCounter[baseName] = 0;
  }
  nameCounter[baseName] += 1;
  return `${baseName} - ${nameCounter[baseName]}`;
}

function drawBrowserSettingTable(tabName, browserConf) {
  tfvis.visor().surface(
      {name: 'browser setting', tab: tabName, styles: {width: '100%'}});
}

function drawBenchmarkParameterTable(tabName) {
  tfvis.visor().surface(
      {name: 'benchmark parameter', tab: tabName, styles: {width: '100%'}});
}

function createTab(browserConf) {
  const tabName = getTabName(browserConf);
  drawBrowserSettingTable(tabName, browserConf);
  drawBenchmarkParameterTable(tabName);

  // TODO: add a 'loading indicator' under the tab.

  return tabName;
}

function reportBenchmarkResults(benchmarkResults) {
  const tabName = benchmarkResults.tabName;

  // TODO:
  //   1. draw a summary table for inference time and memory info.
  //   2. draw a line chart for inference time.
  //   3. draw a table for inference kernel information.
  tfvis.visor().surface(
      {name: 'benchmark results', tab: tabName, styles: {width: '100%'}});

  // TODO: delete 'loading indicator' under the tab.
}

socket.on('benchmarkComplete', benchmarkResult => {
  if (benchmarkResult.error != null) {
    document.getElementById('results').innerHTML += benchmarkResult.error;
  } else {
    reportBenchmarkResults(benchmarkResult);
  }

  // Enable the button.
  benchmarkButton.__li.style.pointerEvents = '';
  benchmarkButton.__li.style.opacity = 1;
});

const gui = new dat.gui.GUI();
gui.domElement.id = 'gui';
showModelSelection();
showParameterSettings();
const benchmarkButton = gui.add(state, 'run').name('Run benchmark');

function showModelSelection() {
  const modelFolder = gui.addFolder('Model');
  let modelUrlController = null;

  modelFolder.add(state.benchmark, 'model', Object.keys(benchmarks))
      .name('model name')
      .onChange(async model => {
        if (model === 'custom') {
          if (modelUrlController === null) {
            modelUrlController = modelFolder.add(state.benchmark, 'modelUrl');
            modelUrlController.domElement.querySelector('input').placeholder =
                'https://your-domain.com/model-path/model.json';
          }
        } else if (modelUrlController != null) {
          modelFolder.remove(modelUrlController);
          modelUrlController = null;
        }
      });
  modelFolder.open();
  return modelFolder;
}

function showParameterSettings() {
  const parameterFolder = gui.addFolder('Parameters');
  parameterFolder.add(state.benchmark, 'numRuns');
  parameterFolder.add(state.benchmark, 'backend', ['wasm', 'webgl', 'cpu']);
  parameterFolder.open();
  return parameterFolder;
}
