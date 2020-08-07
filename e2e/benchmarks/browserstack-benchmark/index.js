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
const socket = io();
const state = {
  isVisorInitiated: false,
  isDatGuiHidden: false,
  browser: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: '84.0',
    os: 'OS X',
    os_version: 'Catalina',
    device: 'null'
  },
  benchmark: {model: 'mobilenet_v2', modelUrl: '', numRuns: 1, backend: 'wasm'},

  run: () => {
    // Disable the button.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = .5;

    initVisor();
    const tabId = createTab(state.browser);

    // Send the benchmark configuration to the server to start the benchmark.
    if (state.browser.device === 'null') {
      state.browser.device = null;
    }
    const benchmark = {...state.benchmark};
    if (state.benchmark.model !== 'custom') {
      delete benchmark['modelUrl'];
    }

    socket.emit('run', {tabId, benchmark, browser: state.browser});
  }
};

function initVisor() {
  if (state.isVisorInitiated) {
    return;
  }
  state.isVisorInitiated = true;

  // Bind an event to visor's 'Maximize/Minimize' button.
  const visorFullScreenButton =
      tfvis.visor().el.getElementsByTagName('button')[0];
  visorFullScreenButton.onclick = () => {
    if (state.isDatGuiHidden) {
      gui.show();
    } else {
      gui.hide();
    }
    state.isDatGuiHidden = !state.isDatGuiHidden;
  };

  // If this button (hide visor) is exposed, then too much extra logics will be
  // needed to tell the full story.
  const visorHideButton = tfvis.visor().el.getElementsByTagName('button')[1];
  visorHideButton.style.display = 'none';
}

const visorTabNameCounter = {};
/**
 *  Generate a unique name for the given setting.
 *
 * @param {object} browserConf An object including os, os_version, browser,
 *     browser_version and device fields.
 */
function getTabId(browserConf) {
  let baseName;
  if (browserConf.os === 'android' || browserConf.os === 'ios') {
    baseName = browserConf.device;
  } else {
    baseName = `${browserConf.os}(${browserConf.os_version})`;
  }
  if (visorTabNameCounter[baseName] == null) {
    visorTabNameCounter[baseName] = 0;
  }
  visorTabNameCounter[baseName] += 1;
  return `${baseName} - ${visorTabNameCounter[baseName]}`;
}

function createTab(browserConf) {
  const tabId = getTabId(browserConf);

  // For tfjs-vis, the tab name is not only a name but also the index to the
  // tab.
  drawBrowserSettingTable(tabId, browserConf);
  drawBenchmarkParameterTable(tabId);

  // TODO: add a 'loading indicator' under the tab.

  return tabId;
}

function reportBenchmarkResults(benchmarkResults) {
  const tabId = benchmarkResults.tabId;

  // TODO: show error message, if `benchmarkResult.error != null`.

  drawBenchmarkResultSummaryTable(benchmarkResults);
  drawInferenceTimeLineChart(benchmarkResults);
  // TODO: draw a table for inference kernel information.

  // TODO: delete 'loading indicator' under the tab.
}

function drawBenchmarkResultSummaryTable(benchmarkResults) {
  const headers = ['Field', 'Value'];
  const values = [];

  const {timeInfo, memoryInfo, tabId} = benchmarkResults;
  const timeArray = benchmarkResults.timeInfo.times;
  const numRuns = timeArray.length;

  if (numRuns >= 1) {
    values.push(['1st inference time', printTime(timeArray[0])]);
    if (numRuns >= 2) {
      values.push(['2nd inference time', printTime(timeArray[1])]);
    }
    values.push([
      `Average inference time (${numRuns} runs)`,
      printTime(timeInfo.averageTime)
    ]);
    values.push(['Best time', printTime(timeInfo.minTime)]);
    values.push(['Worst time', printTime(timeInfo.maxTime)]);
  }

  values.push(['Peak memory', printMemory(memoryInfo.peakBytes)]);
  values.push(['Leaked tensors', memoryInfo.newTensors]);

  values.push(['Number of kernels', memoryInfo.kernels.length]);

  const surface = {
    name: 'Benchmark Result',
    tab: tabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

function drawInferenceTimeLineChart(benchmarkResults) {
  const tabId = benchmarkResults.tabId;
  const inferenceTimeArray = benchmarkResults.timeInfo.times;
  if (inferenceTimeArray.length < 2) {
    return;
  }

  const values = inferenceTimeArray.map((y, x) => ({x, y}));
  const data = {values};
  const surface = {name: 'Inference Time', tab: tabId, styles: {width: '100%'}};
  tfvis.render.linechart(surface, data);
}


function drawBrowserSettingTable(tabId, browserConf) {
  const headers = ['Field', 'Value'];
  const values = [];
  for (const fieldName of TUNABLE_BROWSER_FIELDS) {
    if (browserConf[fieldName] != null && browserConf[fieldName] !== 'null') {
      const row = [fieldName, browserConf[fieldName]];
      values.push(row);
    }
  }
  const surface = {
    name: 'Browser Setting',
    tab: tabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

function drawBenchmarkParameterTable(tabId) {
  const headers = ['Field', 'Value'];
  const values = [];

  for (const entry of Object.entries(state.benchmark)) {
    if (entry[1] != null && entry[1] !== '') {
      values.push(entry);
    }
  }

  const surface = {
    name: 'Benchmark Parameter',
    tab: tabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

socket.on('benchmarkComplete', benchmarkResult => {
  reportBenchmarkResults(benchmarkResult);

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

function printTime(elapsed) {
  return elapsed.toFixed(1) + ' ms';
}

function printMemory(bytes) {
  if (bytes < 1024) {
    return bytes + ' B';
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(2) + ' KB';
  } else {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }
}
