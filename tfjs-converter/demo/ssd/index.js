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
import { CLASSES } from './classes';
import imageURL from './image1.jpg';
import image2URL from './image2.jpg';

const GOOGLE_CLOUD_STORAGE_DIR =
  'https://storage.googleapis.com/tfjs-models/savedmodel/';
const MODEL_URL = GOOGLE_CLOUD_STORAGE_DIR + 'ssd_mobilenet_v1/model.json';

let modelPromise;

window.onload = () => (modelPromise = tf.loadGraphModel(MODEL_URL));

const button = document.getElementById('toggle');
button.onclick = () => {
  image.src = image.src.endsWith(imageURL) ? image2URL : imageURL;
};

const image = document.getElementById('image');
image.src = imageURL;

const buildDetectedObjects = (
  width,
  height,
  boxes,
  scores,
  indexes,
  classes
) => {
  const count = indexes.length;
  const objects = [];
  for (let i = 0; i < count; i++) {
    const bbox = [];
    for (let j = 0; j < 4; j++) {
      bbox[j] = boxes[indexes[i] * 4 + j];
    }
    const minY = bbox[0] * height;
    const minX = bbox[1] * width;
    const maxY = bbox[2] * height;
    const maxX = bbox[3] * width;
    bbox[0] = minX;
    bbox[1] = minY;
    bbox[2] = maxX - minX;
    bbox[3] = maxY - minY;
    objects.push({
      bbox: bbox,
      class: CLASSES[classes[indexes[i]] + 1].displayName,
      score: scores[indexes[i]],
    });
  }
  return objects;
};

const calculateMaxScores = (scores, numBoxes, numClasses) => {
  const maxes = [];
  const classes = [];
  for (let i = 0; i < numBoxes; i++) {
    let max = Number.MIN_VALUE;
    let index = -1;
    for (let j = 0; j < numClasses; j++) {
      if (scores[i * numClasses + j] > max) {
        max = scores[i * numClasses + j];
        index = j;
      }
    }
    maxes[i] = max;
    classes[i] = index;
  }
  return [maxes, classes];
};

/**
 * Infers through the model.
 * @param model The converted model.
 * @param img The image to classify. Can be a tensor or a DOM element image,
 * video, or canvas.
 * @param maxNumBoxes The maximum number of bounding boxes of detected
 * objects. There can be multiple objects of the same class, but at different
 * locations. Defaults to 20.
 * @param minScore The minimum score of the returned bounding boxes
 * of detected objects. Value between 0 and 1. Defaults to 0.5.
 */
const infer = async (model, img, maxNumBoxes, minScore) => {
  const batched = tf.tidy(() => {
    if (!(img instanceof tf.Tensor)) {
      img = tf.browser.fromPixels(img);
    }
    // Reshape to a single-element batch so we can pass it to executeAsync.
    return tf.expandDims(img);
  });
  const height = batched.shape[1];
  const width = batched.shape[2];
  // model returns two tensors:
  // 1. box classification score with shape of [1, 1917, 90]
  // 2. box location with shape of [1, 1917, 1, 4]
  // where 1917 is the number of box detectors, 90 is the number of classes.
  // and 4 is the four coordinates of the box.
  const result = await model.executeAsync(batched);
  const scores = result[0].dataSync();
  const boxes = result[1].dataSync();
  // clean the webgl tensors
  batched.dispose();
  tf.dispose(result);
  const [maxScores, classes] = calculateMaxScores(
    scores,
    result[0].shape[1],
    result[0].shape[2]
  );
  const prevBackend = tf.getBackend();
  // run post process in cpu
  if (tf.getBackend() === 'webgl') {
    tf.setBackend('cpu');
  }
  const indexTensor = tf.tidy(() => {
    const boxes2 = tf.tensor2d(boxes, [result[1].shape[1], result[1].shape[3]]);
    return tf.image.nonMaxSuppression(
      boxes2,
      maxScores,
      maxNumBoxes,
      minScore,
      minScore
    );
  });
  const indexes = indexTensor.dataSync();
  indexTensor.dispose();
  // restore previous backend
  if (prevBackend !== tf.getBackend()) {
    tf.setBackend(prevBackend);
  }
  return buildDetectedObjects(
    width,
    height,
    boxes,
    maxScores,
    indexes,
    classes
  );
};

const runButton = document.getElementById('run');
runButton.onclick = async () => {
  const model = await modelPromise;
  console.log('model loaded');

  console.time('predict1');
  const result = await infer(model, image, 20, 0.5);
  console.timeEnd('predict1');

  const c = document.getElementById('canvas');
  const context = c.getContext('2d');
  context.drawImage(image, 0, 0);
  context.font = '10px Arial';

  console.log('number of detections: ', result.length);
  for (let i = 0; i < result.length; i++) {
    context.beginPath();
    context.rect(...result[i].bbox);
    context.lineWidth = 1;
    context.strokeStyle = 'green';
    context.fillStyle = 'green';
    context.stroke();
    context.fillText(
      result[i].score.toFixed(3) + ' ' + result[i].class,
      result[i].bbox[0],
      result[i].bbox[1] > 10 ? result[i].bbox[1] - 5 : 10
    );
  }
};
