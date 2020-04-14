/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

const MOMENT_DISPLAY_FORMAT = 'MM/DD/YYYY';
const MAX_NUM_LOGS = 50;
const START_LOGGING_DATE = '2019-08-16';
const CHART_HEIGHT = 200;

const containerEl = document.querySelector('#container');
const CHART_WIDTH = containerEl.offsetWidth;

let startDate = moment(START_LOGGING_DATE, 'YYYY-MM-DD'), endDate = moment();

function getLogFiles(start, end) {
  const daysElapsed = end.diff(start, 'd');
  const results = [];
  const formatted = []

  let interval = 1;
  while (daysElapsed / interval > MAX_NUM_LOGS) {
    interval += 1;
  }

  for (let i = 0; i <= daysElapsed; i += interval) {
    const current = endDate.clone().subtract(i, 'days');
    results.unshift(`${current.format('MM_DD_YYYY')}`);
    formatted.unshift(current.format('M/DD'));
  }

  return {results, formatted};
}

function templateTimeSelection(start, end) {
  document.querySelector(".start-date").innerHTML = start.format(MOMENT_DISPLAY_FORMAT);
  document.querySelector(".end-date").innerHTML = end.format(MOMENT_DISPLAY_FORMAT);
}

function closeModal() {
  containerEl.classList.remove("show-modal");
}

function openModal() {
  containerEl.classList.add("show-modal");

  document.querySelector(".editable-start-date").value = startDate.format(MOMENT_DISPLAY_FORMAT);
  document.querySelector(".editable-end-date").value = endDate.format(MOMENT_DISPLAY_FORMAT);
}

function templateBenchmarksForTimePeriod(start, end) {
  const logFiles = getLogFiles(start, end);
  const files = logFiles['results'];

  document.querySelector('.mdl-tabs__tab-bar').innerHTML = '';
  // remove all panels
  [].slice.call(document.querySelectorAll('.mdl-tabs__panel')).forEach(el => {
    el.parentNode.removeChild(el);
  });

  getDataForFiles(files).then(allResponses => {
    const responses = [], dateFormats = [];
    for(let i=0; i<allResponses.length; i++) {
      if(allResponses[i] != null) {
        responses.push(allResponses[i]);
        dateFormats.push(logFiles['formatted'][i]);
      }
    }

    const processedResponses = [];
    const state = {'activeTarget': 0, 'activeTest': 0};

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

    const data = [{name: 'canary', tests: []}];
    const targetIndex = 0; // Hard coded - Canary is the only target for now.

    // populate data
    for (let i = 0; i < processedResponses.length; i++) {
      const response = processedResponses[i];

      for (let idx = 0; idx < response.length; idx++) {
        const {name, params} = response[idx];
        let testIndex =
            data[targetIndex].tests.map(d => d.name).indexOf(name);

        if (testIndex === -1) {
          data[targetIndex].tests.push({name: name, entries: []});
          testIndex = data[targetIndex].tests.length - 1;
        }

        const timestamp = dateFormats[i];
        data[targetIndex].tests[testIndex].entries.push({timestamp, params});
      }
    }

    data.forEach((target, i) => {
      const name = target.name;
      const targetDOMID = `${name}-panel`;

      let tab = document.querySelector(`[href='#${name}']`);
      if (tab == null) {
        tab = document.createElement('a');
        tab.setAttribute('href', '#' + name);
        tab.textContent = name;
        tab.classList.add('mdl-tabs__tab');
      }

      let panel = document.querySelector(targetDOMID);
      if(panel == null) {
        panel = document.createElement('div');
        panel.classList.add('mdl-tabs__panel');
        panel.id = targetDOMID;
      }

      if (i === 0) {
        tab.classList.add('is-active');
        panel.classList.add('is-active');
      }

      target.tests = target.tests
        .filter(test => test.entries.length > 1)
        .sort((a, b) => a.localeCompare(b));

      target.tests.forEach((test, i) => {
          const params = test.entries.reduce((acc, curr) => {
            curr.params.forEach(param => {
              if (acc[param.name] == null) {
                acc[param.name] = [];
              }

              acc[param.name].push({ms: param.ms});
            });
            return acc;
          }, {});

          const msArray = test.entries.map(d => d.params.map(p => p.ms))
                              .reduce((acc, curr) => acc.concat(curr), []);
          const max = Math.max(...msArray);
          const min = 0;

          let increment = 1;
          while ((CHART_WIDTH / ((test.entries.length - 1) / increment)) <
                  20 /* minimum increment width */) {
            increment *= 2;
          }

          const xIncrement = CHART_WIDTH / (test.entries.length - 1);
          const template =  // template trendlines
            `<div class='test'>
              <h4 class='test-name'>${test.name}</h4>
              <div class='legend'>${Object.keys(params).map(param => {
                const backgroundColor =
                getSwatchBackground(swatches[param], strokes[param]);
                return `<div class='swatch'>
                  <div class='color' style='background: ${backgroundColor}'></div>
                  <div class='label'>${param}</div>
                </div>`;
              }).join(' ')}</div>
              <div class='graph-container'>
                <div style='height:${CHART_HEIGHT}px' class='y-axis-labels'>
                  <div class='y-max'>${max}ms</div>
                  <div class='y-min'>${min}ms</div>
                </div>
                <svg data-index=${i} class='graph'
                  width='${CHART_WIDTH}' height='${CHART_HEIGHT}'>
                  ${Object.keys(params).map((param) =>
                    `<path stroke-dasharray='${strokes[param]}'
                        stroke='${swatches[param]}'
                        d='M${params[param].map((d, i) =>
                          `${i * xIncrement},
                          ${CHART_HEIGHT * (1 - ((d.ms - min) / (max - min)))}`)
                        .join('L')}'></path>`)}
                </svg>
                <div class='x-axis-labels'>
                  ${test.entries.map((d, i) => {
                    if (i % increment === 0) {
                      return `<div class='x-label'
                        style='left:${(i / increment) *
                          (CHART_WIDTH / ((test.entries.length - 1) / increment))}px'>
                          ${d.timestamp}</div>`;
                    }
                    return '';
                  }).join(' ')}</div>
                <div class='detail-panel'>
                  <div class='line'></div>
                  <div class='contents'></div>
                </div>
              </div>
            </div>`;

        panel.innerHTML += template;
      });

      document.querySelector('.mdl-tabs__tab-bar').appendChild(tab);
      document.querySelector('.mdl-tabs').appendChild(panel);

      resize();

      document.addEventListener('mousemove', e => {  // handle hovering
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
          parentNode.querySelector('.detail-panel .contents').innerHTML = `${
            entries[entryIndex].params.map(d =>
              `<div class='label-wrapper'>
                <div class='color'
                  style='background:
                    ${getSwatchBackground(swatches[d.name], strokes[d.name])}'>
                </div>
                <div class='label'>${d.ms}</div>
              </div>`).join(' ')}`;
        }
      });
    });
  });
}

document.querySelector(".edit-time-wrapper .instructions").innerHTML = `Enter dates in the format ${MOMENT_DISPLAY_FORMAT}, within the time range ${startDate.format(MOMENT_DISPLAY_FORMAT)} to ${endDate.format(MOMENT_DISPLAY_FORMAT)}.`;

document.querySelector(".time-selection-edit-button").addEventListener('click', openModal);

document.querySelector(".modal-cancel-button").addEventListener("click", closeModal);
document.querySelector(".modal-submit-button").addEventListener("click", () => {
  closeModal();
  startDate = moment(document.querySelector(".editable-start-date").value, MOMENT_DISPLAY_FORMAT);
  endDate = moment(document.querySelector(".editable-end-date").value, MOMENT_DISPLAY_FORMAT);

  templateTimeSelection(startDate, endDate);
  templateBenchmarksForTimePeriod(startDate, endDate);
});
document.querySelector(".modal-backdrop").addEventListener("click", closeModal);

templateTimeSelection(startDate, endDate);
templateBenchmarksForTimePeriod(startDate, endDate);
