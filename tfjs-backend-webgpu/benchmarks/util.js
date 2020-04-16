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

async function getDataForFiles(files) {
  return Promise.all(files.map(d => fetch(
    `https://storage.googleapis.com/learnjs-data/webgpu_benchmark_logs/${
        d}.json`).then(d => d.json()).catch(err => console.log(err))));
}

function templateTimeSelection(start, end) {
  startDateEl.innerHTML = start.format(MOMENT_DISPLAY_FORMAT);
  endDateEl.innerHTML = end.format(MOMENT_DISPLAY_FORMAT);
}

function closeModal() {
  container.classList.remove("show-modal");
}

function openModal(start, end) {
  container.classList.add("show-modal");
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
