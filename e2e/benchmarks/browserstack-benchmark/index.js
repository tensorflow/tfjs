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
  const guiCloseButton = document.getElementsByClassName('close-button')[0];
  const originalGuiWidth = gui.domElement.style.width;

  // The following two bound events are to implemet:
  // - When the visor is minimized, the controlled panel is hidden;
  // - When the visor is maximized, the controlled panel appears;
  gui.domElement.style.width = originalGuiWidth;
  visorFullScreenButton.onclick = () => {
    if (state.isDatGuiHidden) {
      // When opening the controll panel, recover the size.
      gui.open();
      gui.domElement.style.width = originalGuiWidth;
    } else {
      // When closing the controll panel, narrow the size.
      gui.close();
      gui.domElement.style.width = '10%';
    }
    state.isDatGuiHidden = !state.isDatGuiHidden;
  };
  guiCloseButton.onclick = () => {
    if (state.isDatGuiHidden) {
      // When opening the controll panel, recover the size.
      gui.domElement.style.width = originalGuiWidth;
    } else {
      // When closing the controll panel, narrow the size.
      gui.domElement.style.width = '10%';
    }
    tfvis.visor().toggleFullScreen();
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

  // For tfvis, the tab name is not only a name but also the index to the tab.
  drawBrowserSettingTable(tabId, browserConf);
  drawBenchmarkParameterTable(tabId);
  // TODO: add a 'loading indicator' under the tab.

  return tabId;
}

function reportBenchmarkResult(benchmarkResult) {
  const tabId = benchmarkResult.tabId;

  if (benchmarkResult.error != null) {
    // TODO: show error message under the tab.
    alert(benchmarkResult.error);
    return;
  }

  drawInferenceTimeLineChart(benchmarkResult);
  drawBenchmarkResultSummaryTable(benchmarkResult);
  // TODO: draw a table for inference kernel information.
  // This will be done, when we can get kernel timing info from `tf.profile()`.

  // TODO: delete 'loading indicator' under the tab.
}

function drawBenchmarkResultSummaryTable(benchmarkResult) {
  const headers = ['Field', 'Value'];
  const values = [];

  const {timeInfo, memoryInfo, tabId} = benchmarkResult;
  const timeArray = benchmarkResult.timeInfo.times;
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
    name: 'Benchmark Summary',
    tab: tabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

async function drawInferenceTimeLineChart(benchmarkResult) {
  const inferenceTimeArray = benchmarkResult.timeInfo.times;
  if (inferenceTimeArray.length <= 2) {
    return;
  }

  const tabId = benchmarkResult.tabId;
  const values = [];
  inferenceTimeArray.forEach((time, index) => {
    // The first inference time is much larger than other times for webgl,
    // skewing the scaling, so it is removed from the line chart.
    if (index === 0) {
      return;
    }
    values.push({x: index + 1, y: time});
  });

  const surface = {
    name: `2nd - ${inferenceTimeArray.length}st Inference Time`,
    tab: tabId,
    styles: {width: '100%'}
  };
  const data = {values};
  const drawOptions =
      {zoomToFit: true, xLabel: '', yLabel: 'time (ms)', xType: 'ordinal'};

  await tfvis.render.linechart(surface, data, drawOptions);

  // Whenever resize the parent div element, re-draw the chart canvas.
  try {
    const originalCanvasHeight = tfvis.visor()
                                     .surface(surface)
                                     .drawArea.getElementsByTagName('canvas')[0]
                                     .height;
    const labelElement = tfvis.visor().surface(surface).label;

    new ResizeObserver(() => {
      // Keep the height of chart/canvas unchanged.
      tfvis.visor()
          .surface(surface)
          .drawArea.getElementsByTagName('canvas')[0]
          .height = originalCanvasHeight;
      tfvis.render.linechart(surface, data, drawOptions);
    }).observe(labelElement);
  } catch (e) {
    console.warn(`The browser does not support the ResizeObserver API: ${e}`);
  }
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
  // Enable the button.
  benchmarkButton.__li.style.pointerEvents = '';
  benchmarkButton.__li.style.opacity = 1;

  reportBenchmarkResult(benchmarkResult);
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
  return `${elapsed.toFixed(1)} ms`;
}

function printMemory(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
}
