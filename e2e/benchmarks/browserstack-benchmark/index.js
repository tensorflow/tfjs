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
const WAITING_STATUS_COLOR = '#AAAAAA';
const COMPLETE_STATUS_COLOR = '#357edd';
const ERROR_STATUS_COLOR = '#e8564b';
const DISABLED_BUTTON_OPACITY = 0.8;
const ENABLED_BUTTON_OPACITY = 1;
const socket = io();

/**
 * Helps assign unique id for visor tabs.
 * @type {Object<string, number>}
 */
const visorTabNameCounter = {};

const state = {
  isVisorInitiated: false,
  isDatGuiHidden: false,

  // The following `browser` and `benchmark` fields are kept updated by dat.gui.
  browser: {
    base: 'BrowserStack',
    browser: 'chrome',
    browser_version: '84.0',
    os: 'OS X',
    os_version: 'Catalina',
    device: 'null'
  },
  benchmark: {model: 'mobilenet_v2', modelUrl: '', numRuns: 1, backend: 'wasm'},

  /**
   * An array of browser configurations, used to record the browsers to
   * benchmark in this round.
   * @type {!Array<!Object<string, string>>}
   */
  browsers: [],

  /**
   * The id for the visor tab that is used to show the summary for the incoming
   * round of benchmark (has not started).
   * @type {string}
   */
  summaryTabId: getTabId(),

  addBrowser: () => {
    // Add browser config to `state.browsers` array.
    state.browsers.push(state.browser);

    // Enable the benchmark button.
    benchmarkButton.__li.style.pointerEvents = '';
    benchmarkButton.__li.style.opacity = ENABLED_BUTTON_OPACITY;

    // Initialize tfvis, if it is the first call.
    initVisor();
    // (Re-)draw the browser list table, based on the current browsers.
    drawTunableBrowserSummaryTable(state.summaryTabId, state.browsers);
  },

  removeBrowser: index => {
    if (index >= state.browsers.length) {
      throw new Error(
          `Invalid index ${index}, while the state.browsers only ` +
          `has ${state.browsers.length} items.`);
    }

    // Remove the browser from the `state.browsers` array.
    state.browsers.splice(index, 1);

    if (state.browsers.length === 0) {
      // Disable the benchmark button.
      benchmarkButton.__li.style.pointerEvents = 'none';
      benchmarkButton.__li.style.opacity = DISABLED_BUTTON_OPACITY;
    }

    // Re-draw the browser list table, based on the current browsers.
    drawTunableBrowserSummaryTable(state.summaryTabId, state.browsers);
  },

  clearBrowsers: () => {
    state.browsers.splice(0, state.browsers.length);

    // Disable the benchmark button.
    benchmarkButton.__li.style.pointerEvents = 'none';
    benchmarkButton.__li.style.opacity = DISABLED_BUTTON_OPACITY;
  },

  run: () => {
    // Disable the 'Add browser' button.
    addingBrowserButton.__li.style.pointerEvents = 'none';
    addingBrowserButton.__li.style.opacity = DISABLED_BUTTON_OPACITY;

    // Initialize tfvis, if it is the first call.
    initVisor();

    // Build 'tabId - browser config' map (one of benchmark config arguments).
    const browserTabIdConfigMap = {};
    state.browsers.forEach(browser => {
      const tabId = createTab(browser);
      if (browser.device === 'null') {
        browser.device = null;
      }
      browserTabIdConfigMap[tabId] = browser;
    });

    const benchmark = {...state.benchmark};
    if (state.benchmark.model !== 'custom') {
      delete benchmark['modelUrl'];
    }

    // Send the configuration object to the server to start the benchmark.
    socket.emit('run', {
      summaryTabId: state.summaryTabId,
      benchmark,
      browsers: browserTabIdConfigMap,
    });

    // Re-draw the browser list table (untunable) with tabId.
    drawUntunableBrowserSummaryTable(state.summaryTabId, browserTabIdConfigMap);

    // Prepare for the next round benchmark.
    state.summaryTabId = getTabId();
    state.clearBrowsers();
  }
};

let gui;
let benchmarkButton;
let addingBrowserButton;

/**
 * Create a tunable browser list table that can be used to remove browsers.
 *
 * This table is shown before benchmarking.
 *
 * @param {string} summaryTabId The summary tab id.
 * @param {Array<!Object<string, string>>} browsers A list of browsers to
 *     benchmark. The browser object should include fields in
 *     `TUNABLE_BROWSER_FIELDS`.
 */
function drawTunableBrowserSummaryTable(summaryTabId, browsers) {
  const headers = [...TUNABLE_BROWSER_FIELDS, ''];
  const values = [];

  for (let index = 0; index < browsers.length; index++) {
    const browser = browsers[index];
    const row = [];
    for (const fieldName of TUNABLE_BROWSER_FIELDS) {
      if (browser[fieldName] == null || browser[fieldName] === 'null') {
        row.push('-');
      } else {
        row.push(browser[fieldName]);
      }
    }

    // Whenever a browser configuration is removed, this table will be re-drawn,
    // so the index (the argument for state.removeBrowser) will be re-assigned.
    const removeBrowserButtonElement =
        `<button onclick="state.removeBrowser(${index})">Remove</button>`;
    row.push(removeBrowserButtonElement);

    values.push(row);
  }

  const surface = {
    name: 'Browsers to benchmark',
    tab: summaryTabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

/**
 * Create a untunable browser list table, including tab ids. The tab id will be
 * used in the following charts under this summary tab.
 *
 * This table is shown when it is benchmarking or the benchmark is complete.
 *
 * @param {string} summaryTabId The summary tab id.
 * @param {!Object<string, !Object<string, string>>} browserTabIdConfigMap A map
 *     of ' tabId - browser' pairs.
 */
function drawUntunableBrowserSummaryTable(summaryTabId, browserTabIdConfigMap) {
  const headers = ['tabId', ...TUNABLE_BROWSER_FIELDS];
  const values = [];
  for (const browserTabId in browserTabIdConfigMap) {
    const browser = browserTabIdConfigMap[browserTabId];
    const row = [browserTabId];
    for (const fieldName of TUNABLE_BROWSER_FIELDS) {
      row.push(browser[fieldName] || '-');
    }
    values.push(row);
  }

  const surface = {
    name: 'Browsers to benchmark',
    tab: summaryTabId,
    styles: {width: '100%'}
  };
  tfvis.render.table(surface, {headers, values});
}

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

/**
 *  Generate a unique id/name for the given setting. tfvis uses tab name as the
 * index for the tab.
 *
 * @param {?Object<string, string>=} browserConf An object including fields in
 *     `TUNABLE_BROWSER_FIELDS`.
 */
function getTabId(browserConf) {
  let baseName;
  if (browserConf == null) {
    // The tab is a summary tab.
    baseName = 'Summary';
  } else if (browserConf.os === 'android' || browserConf.os === 'ios') {
    // For mobile devices.
    baseName = browserConf.device;
  } else {
    baseName = `${browserConf.os}_${browserConf.os_version}`;
  }
  baseName = baseName.split(' ').join('_');
  if (visorTabNameCounter[baseName] == null) {
    visorTabNameCounter[baseName] = 0;
  }
  visorTabNameCounter[baseName] += 1;
  return `${baseName}_${visorTabNameCounter[baseName]}`;
}

function createTab(browserConf) {
  const tabId = getTabId(browserConf);

  // For tfvis, the tab name is not only a name but also the index to the tab.
  drawBrowserSettingTable(tabId, browserConf);
  drawBenchmarkParameterTable(tabId);

  // Add a status indicator into the tab button.
  const visorTabList = document.getElementsByClassName('tf-tab');
  const curTabElement = visorTabList[visorTabList.length - 1];
  const indicatorElement = document.createElement('span');
  indicatorElement.innerHTML = '.';
  indicatorElement.style.fontSize = '20px';
  indicatorElement.id = `${tabId}-indicator`;
  curTabElement.appendChild(indicatorElement);

  setTabStatus(tabId, 'WAITING');
  addLoaderElement(tabId);
  return tabId;
}

function reportBenchmarkResult(benchmarkResult) {
  const tabId = benchmarkResult.tabId;
  removeLoaderElement(tabId);

  if (benchmarkResult.error != null) {
    setTabStatus(tabId, 'ERROR');
    // TODO: show error message under the tab.
    console.log(benchmarkResult.error);
  } else {
    setTabStatus(tabId, 'COMPLETE');
    drawInferenceTimeLineChart(benchmarkResult);
    drawBenchmarkResultSummaryTable(benchmarkResult);
    // TODO: draw a table for inference kernel information.
    // This will be done, when we can get kernel timing info from
    // `tf.profile()`.
  }
}

/**
 * Set the status for the given tab. The status can be 'WAITING', 'COMPLETE' or
 * 'ERROR'.
 *
 * @param {string} tabId  The index element id of the tab.
 * @param {string} status The status to be set for the tab.
 */
function setTabStatus(tabId, status) {
  const indicatorElementId = `${tabId}-indicator`;
  const indicatorElement = document.getElementById(indicatorElementId);
  switch (status) {
    case 'WAITING':
      indicatorElement.style.color = WAITING_STATUS_COLOR;
      break;
    case 'COMPLETE':
      indicatorElement.style.color = COMPLETE_STATUS_COLOR;
      break;
    case 'ERROR':
      indicatorElement.style.color = ERROR_STATUS_COLOR;
      break;
    default:
      throw new Error(`Undefined status: ${status}.`);
  }
}

/**
 * Add a loader element under the tab page.
 *
 * @param {string} tabId
 */
function addLoaderElement(tabId) {
  const surface = tfvis.visor().surface(
      {name: 'Benchmark Summary', tab: tabId, styles: {width: '100%'}});
  const loaderElement = document.createElement('div');
  loaderElement.className = 'loader';
  loaderElement.id = `${tabId}-loader`;
  surface.drawArea.appendChild(loaderElement);
}

/**
 * Remove the loader element under the tab page.
 *
 * @param {string} tabId
 */
function removeLoaderElement(tabId) {
  const loaderElementId = `${tabId}-loader`;
  const loaderElement = document.getElementById(loaderElementId);
  if (loaderElement != null) {
    loaderElement.remove();
  }
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

function onPageLoad() {
  gui = new dat.gui.GUI();
  gui.domElement.id = 'gui';
  showModelSelection();
  showParameterSettings();
  addingBrowserButton = gui.add(state, 'addBrowser').name('Add browser');
  benchmarkButton = gui.add(state, 'run').name('Run benchmark');

  // Disable the 'Run benchmark' button until a browser is added.
  benchmarkButton.__li.style.pointerEvents = 'none';
  benchmarkButton.__li.style.opacity = DISABLED_BUTTON_OPACITY;

  socket.on('benchmarkComplete', benchmarkResult => {
    // Enable the 'Add browser' button.
    addingBrowserButton.__li.style.pointerEvents = '';
    addingBrowserButton.__li.style.opacity = ENABLED_BUTTON_OPACITY;

    reportBenchmarkResult(benchmarkResult);

    // TODO: We can add a summary for the benchmark results in this round.
  });
}
