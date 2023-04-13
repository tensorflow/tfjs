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
const MODEL_URL = GOOGLE_CLOUD_STORAGE_DIR + 'ssdlite_mobilenet_v2/model.json';

let modelPromise;

window.onload = () => modelPromise = tf.loadGraphModel(MODEL_URL);

const button = document.getElementById('toggle');
button.onclick = () => {
  image.src = image.src.endsWith(imageURL) ? image2URL : imageURL;
};

const image = document.getElementById('image');
image.src = imageURL;

const runButton = document.getElementById('run');
runButton.onclick = async () => {
  const model = await modelPromise;
  const pixels = tf.browser.fromPixels(image);
  console.log('model loaded');
  console.time('predict1');
  const res1 = await model.executeAsync(pixels.reshape([1, ...pixels.shape]));
  res1.map(t => t.dataSync());
  console.timeEnd('predict1');
  console.time('predict2');
  const res2 = await model.executeAsync(pixels.reshape([1, ...pixels.shape]));
  const boxes = res2[0].dataSync();
  const scores = res2[1].dataSync();
  console.timeEnd('predict2');
};
