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

const mobilenet = require('@tensorflow-models/mobilenet');
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const jpeg = require('jpeg-js');

const backendNodeGL = require('./../dist/index');

const gl = tf.backend().getGPGPUContext().gl;
console.log(`  - gl.VERSION: ${gl.getParameter(gl.VERSION)}`);
console.log(`  - gl.RENDERER: ${gl.getParameter(gl.RENDERER)}`);

const NUMBER_OF_CHANNELS = 3;
const PREPROCESS_DIVISOR = 255 / 2;

function readImageAsJpeg(path) {
  return jpeg.decode(fs.readFileSync(path), true);
}

function imageByteArray(image, numChannels) {
  const pixels = image.data;
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);
  for (let i = 0; i < numPixels; i++) {
    for (let j = 0; j < numChannels; j++) {
      values[i * numChannels + j] = pixels[i * 4 + j];
    }
  }
  return values;
}

function imageToInput(image, numChannels) {
  const values = imageByteArray(image, numChannels);
  const outShape = [1, image.height, image.width, numChannels];
  const input = tf.tensor4d(values, outShape, 'float32');
  return tf.div(tf.sub(input, PREPROCESS_DIVISOR), PREPROCESS_DIVISOR);
}

async function run(path) {
  const image = readImageAsJpeg(path);
  const input = imageToInput(image, NUMBER_OF_CHANNELS);

  console.log('  - Loading model...');
  let start = tf.util.now();
  const model = await mobilenet.load();
  let end = tf.util.now();
  console.log(`  - Mobilenet load: ${end - start}ms`);

  start = tf.util.now();
  console.log('  - Coldstarting model...');
  await model.classify(input);
  end = tf.util.now();
  console.log(`  - Mobilenet cold start: ${end - start}ms`);

  const times = 100;
  let totalMs = 0;
  console.log(`  - Running inference (${times}x) ...`);
  for (let i = 0; i < times; i++) {
    start = tf.util.now();
    await model.classify(input);
    end = tf.util.now();

    totalMs += end - start;
  }

  console.log(`  - Mobilenet inference: (${times}x) : ${(totalMs / times)}ms`);
}

if (process.argv.length !== 3) {
  throw new Error(
      'incorrect arguments: node packaged-mobilenet-test.js <IMAGE_FILE>');
}

run(process.argv[2]);
