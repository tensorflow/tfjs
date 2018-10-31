/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import 'babel-polyfill';
import * as tf from '@tensorflow/tfjs';
import {CLASSES} from './classes';
import imageURL from './image1.jpg';
import image2URL from './image2.jpg';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/tfjs-models/savedmodel/';
const MODEL_URL =
    GOOGLE_CLOUD_STORAGE_DIR + 'coco-ssd-mobilenet_v1/tensorflowjs_model.pb';
const WEIGHTS_URL =
    GOOGLE_CLOUD_STORAGE_DIR + 'coco-ssd-mobilenet_v1/weights_manifest.json';

let modelPromise;

window.onload = () => modelPromise = tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);

const button = document.getElementById('toggle');
button.onclick = () => {
  image.src = image.src.endsWith(imageURL) ? image2URL : imageURL;
};

const image = document.getElementById('image');
image.src = imageURL;

const runButton = document.getElementById('run');
runButton.onclick = async () => {
  const model = await modelPromise;
  const pixels = tf.fromPixels(image);
  console.log('model loaded');
  console.time('predict1');
  const res1 = await model.executeAsync(pixels.reshape([1, ...pixels.shape]));
  res1.map(t => t.dataSync());
  console.timeEnd('predict1');
  console.time('predict2');
  const res2 = await model.executeAsync(pixels.reshape([1, ...pixels.shape]));
  const count = res2[3].dataSync()[0];
  const boxes = res2[0].dataSync();
  const scores = res2[1].dataSync();
  const classes = res2[2].dataSync();
  console.timeEnd('predict2');


  const c = document.getElementById('canvas');
  const context = c.getContext('2d');
  context.drawImage(image, 0, 0);
  context.font = '10px Arial';

  console.log('number of detections: ', count);
  for (let i = 0; i < count; i++) {
    const min_y = boxes[i * 4] * 399;
    const min_x = boxes[i * 4 + 1] * 600;
    const max_y = boxes[i * 4 + 2] * 399;
    const max_x = boxes[i * 4 + 3] * 600;

    context.beginPath();
    context.rect(min_x, min_y, max_x - min_x, max_y - min_y);
    context.lineWidth = 1;
    context.strokeStyle = 'black';
    context.stroke();
    context.fillText(
        scores[i].toFixed(3) + ' ' + CLASSES.find(label => label.id === classes[i]).display_name,
        min_x, min_y - 5);
  }
};

