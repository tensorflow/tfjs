/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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


function isConv2D(kernel) {
  return kernel.name.includes('Conv2D');
}

function isNotDepthwise(kernel) {
  return !kernel.name.includes('Depthwise');
}

function is1x1Filter(kernel) {
  return kernel.inputShapes.length >= 2 && kernel.inputShapes[1].length === 4 &&
      kernel.inputShapes[1][0] === 1 && kernel.inputShapes[1][1] === 1;
}

async function getKernels(modelKey) {
  const benchmark = benchmarks[modelKey];
  let kernels = [];
  if (benchmark.architectures != null) {
    for (const arch of benchmark.architectures) {
      const model = await benchmark.load(undefined, arch);
      const predict = benchmark.predictFunc();
      const profileInfo =
          await profileInference(() => predict(model), false, 1);
      kernels.push(...profileInfo.kernels);
    }
  } else {
    const model = await benchmark.load();
    const predict = benchmark.predictFunc();
    const profileInfo = await profileInference(() => predict(model), false, 1);
    kernels = profileInfo.kernels;
  }
  return kernels;
}

/**
 * Create a table element, with table head and caption.
 */
function initializeTable(tableName, tableHeaderfileds) {
  const tableElem = document.createElement('TABLE');
  let header = tableElem.createTHead();
  let row = header.insertRow(0);
  row.insertCell(0);
  for (let filedIndex in tableHeaderfileds) {
    let cell = row.insertCell(row.cells.length);
    cell.innerHTML = `<b>${tableHeaderfileds[filedIndex]}</b>`;
  }
  tableElem.createCaption().innerHTML = tableName;
  tableElem.setAttribute('border', '1');
  tableElem.setAttribute('id', tableName);
  return tableElem;
}

function presentKernels(kernels) {
  console.log(kernels.map(e => e.name));
}

function aggregate(items) {
  const map = {};
  for (const item of items) {
    let count = map[item];
    if (count != null) {
      map[item] = count + 1;
    } else {
      map[item] = 1;
    }
  }

  const res = [];
  for (const key of Object.keys(map)) {
    res.push([key, map[key]]);
  }
  return res.sort((x, y) => y[1] - x[1]);
}

function aggregatedArrayToStr(aggregatedArray) {
  const total =
      aggregatedArray.map(e => e[1]).reduce((pre, cur) => pre + cur, 0);
  return aggregatedArray.slice(0, showTopNum)
      .map(e => `${e[0]}   (cout:${e[1]}, ${(e[1] / total * 100).toFixed(2)}%)`)
      .join('<br/>');
}

function presentPointwiseKernels(tableElem, name, kernels) {
  // Check kernels
  kernels.forEach(kernel => {
    if (kernel.inputShapes.length < 2 || kernel.inputShapes[0].length !== 4 ||
        kernel.inputShapes[1].length !== 4 ||
        kernel.inputShapes[0][3] !==
            kernel.inputShapes[1][2]) {  // Channel last.
      console.warn(kernel);
    }
  });

  const row = tableElem.insertRow(tableElem.rows.length);
  let nameCell = row.insertCell(0);
  nameCell.innerHTML = name;

  const cell1 = row.insertCell(1);
  const imageWidthAgg = aggregate(kernels.map(e => e.inputShapes[0][1]));
  cell1.innerHTML = aggregatedArrayToStr(imageWidthAgg);

  const cell2 = row.insertCell(2);
  const imageShapeAgg = aggregate(
      kernels.map(e => `${e.inputShapes[0][1]}x${e.inputShapes[0][2]}`));
  cell2.innerHTML = aggregatedArrayToStr(imageShapeAgg);

  const cell3 = row.insertCell(3);
  const inputChannelAgg = aggregate(kernels.map(e => e.inputShapes[0][3]));
  cell3.innerHTML = aggregatedArrayToStr(inputChannelAgg);

  const cell4 = row.insertCell(4);
  const outputChannelAgg = aggregate(kernels.map(e => e.inputShapes[1][3]));
  cell4.innerHTML = aggregatedArrayToStr(outputChannelAgg);

  const cell5 = row.insertCell(5);
  const imgAgg = aggregate(kernels.map(e => `[${e.inputShapes[0]}]`));
  cell5.innerHTML = aggregatedArrayToStr(imgAgg);

  const cell6 = row.insertCell(6);
  const filterAgg = aggregate(kernels.map(e => `[${e.inputShapes[1]}]`));
  cell6.innerHTML = aggregatedArrayToStr(filterAgg);

  const cell7 = row.insertCell(7);
  const opAgg = aggregate(
      kernels.map(e => `Img:${e.inputShapes[0]}, Filter:${e.inputShapes[1]}`));
  cell7.innerHTML = aggregatedArrayToStr(opAgg);
}
