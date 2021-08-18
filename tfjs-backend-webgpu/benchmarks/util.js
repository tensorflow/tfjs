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

function getLogFiles(start, end) {
  const daysElapsed = end.diff(start, 'd');
  const results = [];
  const formatted = [];
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

async function getDataForFiles(files) {
  return Promise.all(files.map(
      d =>
          fetch(
              `https://storage.googleapis.com/learnjs-data/webgpu_benchmark_logs/${
                  d}.json`)
              .then(d => d.json())
              .catch(err => console.log(err))));
}

function templateTimeSelection(start, end) {
  startDateEl.innerHTML = start.format(MOMENT_DISPLAY_FORMAT);
  endDateEl.innerHTML = end.format(MOMENT_DISPLAY_FORMAT);
}

function closeModal() {
  container.classList.remove('show-modal');
}

function openModal(start, end) {
  container.classList.add('show-modal');
  startDateInput.value = start.format(MOMENT_DISPLAY_FORMAT);
  endDateInput.value = end.format(MOMENT_DISPLAY_FORMAT);
}

function clearDisplay() {
  tabsContainer.innerHTML = '';
  // remove all panels
  [].slice.call(document.querySelectorAll('.mdl-tabs__panel')).forEach(el => {
    el.parentNode.removeChild(el);
  });
}

function getOrCreateTab(name) {
  let tab = document.querySelector(`[href='#${name}']`);
  if (tab == null) {
    tab = document.createElement('a');
    tab.setAttribute('href', '#' + name);
    tab.textContent = name;
    tab.classList.add('mdl-tabs__tab');
  }
  return tab;
}

function getOrCreatePanel(id) {
  let panel = document.querySelector(id);
  if (panel == null) {
    panel = document.createElement('div');
    panel.classList.add('mdl-tabs__panel');
    panel.id = id;
  }
  return panel;
}

function flatten(arr) {
  return arr.reduce((acc, curr) => acc.concat(curr), []);
}

function getIncrementForWidth(width, length, minWidth) {
  let increment = 1;
  while ((width / ((length - 1) / increment)) < minWidth) {
    increment *= 2;
  }
  return increment;
}

function getTrendlinesHTML(test, params, max, increment, xIncrement, i) {
  return `<div class='test'>
    <h4 class='test-name'>${test.name}</h4>
    <div class='legend'>${Object.keys(params).map(param => {
      const backgroundColor =
          getSwatchBackground(SWATCHES[param], STROKES[param]);
      return `<div class='swatch'>
        <div class='color' style='background: ${backgroundColor}'></div>
        <div class='label'>${param}</div>
      </div>`;}).join(' ')}</div>
    <div class='graph-container'>
      <div style='height:${CHART_HEIGHT}px' class='y-axis-labels'>
        <div class='y-max'>${max}ms</div>
        <div class='y-min'>0ms</div>
      </div>
      <svg data-index=${i} class='graph'
        width='${CHART_WIDTH}' height='${CHART_HEIGHT}'>
        ${Object.keys(params).map((param) =>
          `<path stroke-dasharray='${STROKES[param]}'
              stroke='${SWATCHES[param]}'
              d='M${params[param].map((d, i) =>
                `${i * xIncrement}, ${CHART_HEIGHT * (1 - (d.ms / max))}`)
                    .join('L')}'></path>`)}
      </svg>
      <div class='x-axis-labels'>
        ${test.entries.map((d, i) => {
          if (i % increment === 0) {
            const left = (i / increment) *
              (CHART_WIDTH / ((test.entries.length - 1) / increment));
            return `<div class='x-label' style='left:${left}px'>
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
}
