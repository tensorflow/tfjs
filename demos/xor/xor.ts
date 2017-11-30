/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
'use strict';

import '../demo-header';
import '../demo-footer';

import {learnXOR} from './learn-xor';

Polymer({is: 'xor-demo', properties: {}});

const trainButton = document.getElementById('train') as HTMLButtonElement;
const resultDiv = document.getElementById('result') as HTMLDivElement;
const metainfoDiv = document.getElementById('metainfo') as HTMLDivElement;

interface ResultInterface {
  iterations: number;
  loss: number;
  time: number;
  result: Array < {
    input: Float32Array|Int32Array|Uint8Array;
    expected: Float32Array|Int32Array|Uint8Array;
    output: Float32Array|Int32Array|Uint8Array;
  }
  > ;
}

const round = (num: number) => {
  return Math.round(num * 1000) / 1000;
};

const buildResultHTML = (result: ResultInterface) => {
  let html = ``;
  const resultArray = result['result'];
  const correct = `<i class="fa fa-check green" aria-hidden="true"></i>`;
  const incorrect = `<i class="fa fa-times red" aria-hidden="true"></i>`;

  for (const row of resultArray) {
    const input = row['input'];
    const expected = row['expected'];
    const output = row['output'];
    let symbol;

    if (Math.abs(expected[0] - output[0]) < 0.5) {
      symbol = correct;
    } else {
      symbol = incorrect;
    }

    html += `
      <div class="output">
        <b>input</b>
        <span class="dark">${round(input[0])} ${round(input[1])}</span>
        <b>output</b>
        <span class="dark">${round(output[0])}</span>
        <b>expected</b>
        <span class="dark">${round(expected[0])}</span>
        ${symbol}
      </div>
    `;
  }

  return html;
};

const buildMetaInfoHTML = (result: ResultInterface) => {
  return `
    <b>Iterations</b>: ${result['iterations']}  -
    <b>Log loss</b>: ${round(result['loss'])}  -
    <b>Time taken</b>: ${round(result['time'])} ms
  `;
};

const hideDiv = (div: HTMLDivElement) => {
  div.classList.remove('show');
  div.classList.add('hide');
};

const showDiv = (div: HTMLDivElement) => {
  div.classList.remove('hide');
  div.classList.add('show');
};

trainButton.addEventListener('click', () => {
  hideDiv(resultDiv);
  hideDiv(metainfoDiv);

  learnXOR()
      .then((result) => {
        const resultHTML = buildResultHTML(result);
        resultDiv.innerHTML = resultHTML;

        const metainfoHTML = buildMetaInfoHTML(result);
        metainfoDiv.innerHTML = metainfoHTML;

        showDiv(resultDiv);
        showDiv(metainfoDiv);
      })
      .catch((err) => {
        console.log(err);
      });
});
