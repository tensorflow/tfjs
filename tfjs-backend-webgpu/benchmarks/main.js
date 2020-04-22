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

// Create handles to DOM elements.
const container = document.querySelector('#container');
const tabsContainer = document.querySelector('.mdl-tabs__tab-bar');
const tabs = document.querySelector('.mdl-tabs');
const timeSelectionInstructions =
    document.querySelector('.edit-time-wrapper .instructions');
const editTimeButton = document.querySelector('.time-selection-edit-button');
const cancelEditTimeButton = document.querySelector('.modal-cancel-button');
const submitEditTimeButton = document.querySelector('.modal-submit-button');
const startDateEl = document.querySelector('.start-date');
const endDateEl = document.querySelector('.end-date');
const startDateInput = document.querySelector('.editable-start-date');
const endDateInput = document.querySelector('.editable-end-date');
const modalBackdrop = document.querySelector('.modal-backdrop');

const CHART_WIDTH = container.offsetWidth;
const state = {
  'activeTarget': 0,
  'activeTest': 0
};

let startDate = moment(START_LOGGING_DATE, 'YYYY-MM-DD'),
  endDate = moment(),
  graphOffsetLeft = 0,
  data = [];

function resize() {
  graphOffsetLeft = document.querySelector('.graph-container').offsetLeft;
};
window.addEventListener('resize', resize);

function templateBenchmarksForTimePeriod(start, end) {
  const logFiles = getLogFiles(start, end);
  const files = logFiles['results'];

  clearDisplay();

  getDataForFiles(files).then(allResponses => {
    const responses = [], dateFormats = [];
    for (let i = 0; i < allResponses.length; i++) {
      if (allResponses[i] != null) {
        responses.push(allResponses[i]);
        dateFormats.push(logFiles['formatted'][i]);
      }
    }

    const processedResponses = [];
    for (let i = 0; i < responses.length; i++) {
      const response = responses[i];
      const processedResponse = [];

      for (let idx = 0; idx < response.length; idx++) {
        const {name, backend, min, mean} = response[idx];
        let testIndex = processedResponse.map(d => d.name).indexOf(name);

        if (testIndex === -1) {
          processedResponse.push({name: name, params: []});
          testIndex = processedResponse.length - 1;
        }

        processedResponse[testIndex].params.push(
            {name: `${backend}_min`, ms: min});
        processedResponse[testIndex].params.push(
            {name: `${backend}_mean`, ms: mean});
      }
      processedResponses.push(processedResponse);
    }

    data = TARGETS.map(name => ({name, tests: []}));
    const targetIndex = 0;  // Hard coded - Canary is the only target for now.

    // populate data
    for (let i = 0; i < processedResponses.length; i++) {
      const response = processedResponses[i];

      for (let idx = 0; idx < response.length; idx++) {
        const {name, params} = response[idx];
        let testIndex = data[targetIndex].tests.map(d => d.name).indexOf(name);

        if (testIndex === -1) {
          data[targetIndex].tests.push({name: name, entries: []});
          testIndex = data[targetIndex].tests.length - 1;
        }

        const timestamp = dateFormats[i];
        data[targetIndex].tests[testIndex].entries.push({timestamp, params});
      }
    }

    data.forEach((target, i) => {
      const tab = getOrCreateTab(target.name);
      const panel = getOrCreatePanel(`${name}-panel`);
      if (i === 0) {
        tab.classList.add('is-active');
        panel.classList.add('is-active');
      }

      target.tests = target.tests.filter(test => test.entries.length > 1)
                         .sort((a, b) => a.name.localeCompare(b.name));

      target.tests.forEach((test, i) => {
        const params = {};
        test.entries.forEach(entry => {
          entry.params.forEach(({name, ms}) => {
            if (params[name] == null) {
              params[name] = [];
            }

            params[name].push({ms})
          });
        });

        const msArray = flatten(test.entries.map(d => d.params.map(p => p.ms)));
        const max = Math.max(...msArray);
        const increment = getIncrementForWidth(
            CHART_WIDTH, test.entries.length, 20 /* minimum increment width */);
        const xIncrement = CHART_WIDTH / (test.entries.length - 1);
        panel.innerHTML +=
            getTrendlinesHTML(test, params, max, increment, xIncrement, i);
      });

      tabsContainer.appendChild(tab);
      tabs.appendChild(panel);

      resize();
    });
  });
}

document.addEventListener('mousemove', e => {
  if (e.target.classList.contains('graph')) {
    state.activeTest = +e.target.getAttribute('data-index');

    const entries =
        data[state.activeTarget].tests[state.activeTest].entries;
    const left = e.clientX - graphOffsetLeft;
    const entryIndex = Math.max(
        0,
        Math.min(
            entries.length - 1,
            Math.floor((left / CHART_WIDTH) * entries.length)));

    const parentNode = e.target.parentNode;
    parentNode.querySelector('.detail-panel').style.left = left + 'px';
    parentNode.querySelector('.detail-panel .contents').innerHTML =
      `${entries[entryIndex].params.map(d =>
        `<div class='label-wrapper'>
          <div class='color'
            style='background:
              ${getSwatchBackground(SWATCHES[d.name], STROKES[d.name])}'>
          </div>
          <div class='label'>${d.ms}</div>
        </div>`).join(' ')}`;
  }
});

timeSelectionInstructions.innerHTML = `Enter dates in the format <span>${
    MOMENT_DISPLAY_FORMAT}</span>, within the time range <span>${
    startDate.format(MOMENT_DISPLAY_FORMAT)}</span> to <span>${
    endDate.format(MOMENT_DISPLAY_FORMAT)}</span>.`;

editTimeButton.addEventListener('click', () => openModal(startDate, endDate));

cancelEditTimeButton.addEventListener('click', closeModal);
submitEditTimeButton.addEventListener('click', () => {
  closeModal();
  startDate = moment(startDateInput.value, MOMENT_DISPLAY_FORMAT);
  endDate = moment(endDateInput.value, MOMENT_DISPLAY_FORMAT);

  templateTimeSelection(startDate, endDate);
  templateBenchmarksForTimePeriod(startDate, endDate);
});
modalBackdrop.addEventListener('click', closeModal);

templateTimeSelection(startDate, endDate);
templateBenchmarksForTimePeriod(startDate, endDate);
