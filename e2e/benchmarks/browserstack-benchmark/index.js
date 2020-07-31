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

socket.on('benchmarkComplete', benchmarkResult => {
  const {timeInfo, memoryInfo} = benchmarkResult;
  document.getElementById('results').innerHTML +=
      JSON.stringify(timeInfo, null, 2);

  // Enable the button.
  benchmarkButton.__li.style.pointerEvents = '';
  benchmarkButton.__li.style.opacity = 1;
});

const gui = new dat.gui.GUI();
const benchmarkButton = gui.add(state, 'run').name('Run benchmark');
