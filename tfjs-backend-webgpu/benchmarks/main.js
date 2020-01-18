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

const swatches = {
  'webgpu_min': '#F1523E',
  'webgpu_mean': '#F1523E',
  'webgl_min': '#3f51b5',
  'webgl_mean': '#3f51b5'
};

const strokes = {
  'webgpu_min': '2',
  'webgpu_mean': '0',
  'webgl_min': '2',
  'webgl_mean': '0'
};

const MAX_NUM_LOGS = 50;
const START_LOGGING_DATE = '2019-08-16';
const startDate = moment(START_LOGGING_DATE, 'YYYY-MM-DD');
const endDate = moment();
const files = [];
let dateFormats = [];
const daysElapsed = endDate.diff(startDate, 'd');
let interval = 1;

while (daysElapsed / interval > MAX_NUM_LOGS) {
  interval += 1;
}

for (let i = daysElapsed; i >= 0; i -= interval) {
  const current = startDate.clone().add(i, 'days');
  files.unshift(`${current.format('MM_DD_YYYY')}`);
  dateFormats.unshift(current.format('M/DD'));
}

function getSwatchBackground(swatch, stroke) {
  let background = swatch;
  if (stroke > 0) {
    background = `repeating-linear-gradient(
      to right,
      ${swatch},
      ${swatch} 2px,
      white 2px,
      white 4px
    );`;
  }
  return background;
}

Promise
    .all(files.map(
        d =>
            fetch(
                `https://storage.googleapis.com/learnjs-data/webgpu_benchmark_logs/${
                    d}.json`)
                .then(d => d.json())
                .catch(err => console.log(err))))
    .then(responses => {
      dateFormats = dateFormats.filter((d, i) => responses[i] != null);

      const processedResponses = [];

      const state = {'activeTarget': 0, 'activeTest': 0};

      for (let i = 0; i < responses.length; i++) {
        const response = responses[i];
        if (response == null) continue;

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
      // hard coded - only one target for now
      const targetIndex = 0;

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

      const chartHeight = 200;
      const chartWidth = document.querySelector('#container').offsetWidth;

      data.forEach((target, i) => {
        const name = target.name;
        const tab = document.createElement('a');
        tab.setAttribute('href', '#' + name);
        tab.textContent = name;
        tab.classList.add('mdl-tabs__tab');

        const panel = document.createElement('div');
        panel.classList.add('mdl-tabs__panel');
        panel.id = `${name}-panel`;

        if (i === 0) {
          tab.classList.add('is-active');
          panel.classList.add('is-active');
        }

        target.tests.filter(test => test.entries.length > 1)
            .forEach((test, i) => {
              const params = test.entries.reduce((acc, curr) => {
                curr.params.forEach(param => {
                  if (typeof acc[param.name] === 'undefined') {
                    acc[param.name] = [];
                  }

                  acc[param.name].push({ms: param.ms});
                });

                return acc;
              }, {});

              const msArray = test.entries.map(d => d.params.map(p => p.ms))
                                  .reduce((acc, curr) => acc.concat(curr), []);
              const max = Math.max(...msArray);
              // const min = Math.min(...msArray);
              const min = 0;

              const minWidthOfIncrement = 20;
              let increment = 1;
              while ((chartWidth / ((test.entries.length - 1) / increment)) <
                     minWidthOfIncrement) {
                increment *= 2;
              }

              const xIncrement = chartWidth / (test.entries.length - 1);
              const template =  // template trendlines
                  `<div class='test'>
            <h4 class='test-name'>${test.name}</h4>
            <div class='legend'>${
                      Object.keys(params)
                          .map((param, i) => {
                            return `<div class='swatch'>
                                <div class='color' style='background:
                                ${
                                getSwatchBackground(
                                    swatches[param],
                                    strokes
                                        [param])}'></div> <div class='label'>${
                                param}</div>
                              </div>`;
                          })
                          .join(' ')}</div>
            <div class='graph-container'>
              <div style='height:${chartHeight}px' class='y-axis-labels'>
                <div class='y-max'>${max}ms</div>
                <div class='y-min'>${min}ms</div>
              </div>
              <svg data-index=${i} class='graph' width='${
                      chartWidth}' height='${chartHeight}'>${
                      Object.keys(params).map(
                          (param, i) => `<path stroke-dasharray='${
                              strokes[param]}' stroke='${
                              swatches[param]}' d='M${
                              params[param]
                                  .map(
                                      (d, i) => `${i * xIncrement},${
                                          chartHeight *
                                          (1 - ((d.ms - min) / (max - min)))}`)
                                  .join('L')}'></path>`)}</svg>
              <div class='x-axis-labels'>${
                      test.entries
                          .map((d, i) => {
                            if (i % increment === 0) {
                              return `<div class='x-label' style='left:${
                                  (i / increment) *
                                  (chartWidth /
                                   ((test.entries.length - 1) /
                                    increment))}px'>${d.timestamp}</div>`;
                            }
                            return '';
                          })
                          .join(' ')}</div>
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

        let graphOffsetLeft = 0;

        function resize() {
          graphOffsetLeft =
              document.querySelector('.graph-container').offsetLeft;
        }

        window.addEventListener('resize', resize);
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
                    Math.floor((left / chartWidth) * entries.length)));
            const parentNode = e.target.parentNode;
            parentNode.querySelector('.detail-panel').style.left = left + 'px';
            parentNode.querySelector('.detail-panel .contents').innerHTML = `${
                entries[entryIndex]
                    .params
                    .map(
                        (d, i) => {return `<div class='label-wrapper'>
                          <div class='color' style='background: ${
                            getSwatchBackground(
                                swatches[d.name], strokes[d.name])}'></div>
                          <div class='label'>${d.ms}</div>
                        </div>`})
                    .join(' ')}
            `;
          }
        });
      });
    });
