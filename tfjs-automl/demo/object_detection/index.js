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

import '@tensorflow/tfjs-backend-webgl';
import * as automl from '@tensorflow/tfjs-automl';

const MODEL_URL =
    'https://storage.googleapis.com/tfjs-testing/tfjs-automl/object_detection/model.json';

async function run() {
  const model = await automl.loadObjectDetection(MODEL_URL);
  const image = document.getElementById('salad');
  // These are the default options.
  const options = {score: 0.5, iou: 0.5, topk: 20};
  const predictions = await model.detect(image, options);

  // Show the resulting object on the page.
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(predictions, null, 2);
  document.body.append(pre);

  drawBoxes(predictions);
}

// Overlays boxes with labels onto the image using `rect` and `text` svg
// elements.
function drawBoxes(predictions) {
  const svg = document.querySelector('svg');
  predictions.forEach(prediction => {
    const {box, label, score} = prediction;
    const {left, top, width, height} = box;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', width);
    rect.setAttribute('height', height);
    rect.setAttribute('x', left);
    rect.setAttribute('y', top);
    rect.setAttribute('class', 'box');
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', left + width / 2);
    text.setAttribute('y', top);
    text.setAttribute('dy', 12);
    text.setAttribute('class', 'label');
    text.textContent = `${label}: ${score.toFixed(3)}`;
    svg.appendChild(rect);
    svg.appendChild(text);
    const textBBox = text.getBBox();
    const textRect =
        document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    textRect.setAttribute('x', textBBox.x);
    textRect.setAttribute('y', textBBox.y);
    textRect.setAttribute('width', textBBox.width);
    textRect.setAttribute('height', textBBox.height);
    textRect.setAttribute('class', 'label-rect');
    svg.insertBefore(textRect, text);
  });
}

run();
