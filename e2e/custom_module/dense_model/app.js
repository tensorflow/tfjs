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

import * as tf from '@tensorflow/tfjs';

async function main(modelUrl) {
  await tf.ready();
  const backend = tf.getBackend();
  self.postMessage({
    msg: true,
    payload: `Backend ready: ${backend}. Got model url: ${modelUrl}`
  });

  const registeredKernels = tf.getKernelsForBackend(backend);
  let model;
  try {
    model = await tf.loadGraphModel(modelUrl)
  } catch (e) {
    self.postMessage({error: true, payload: e});
  }

  const predictions = model.predict(tf.tensor2d([20], [1, 1])).dataSync();
  // result is 38.17822265625

  // send the final result of the test.
  self.postMessage({
    result: true,
    payload: {
      numKernels: registeredKernels.length,
      kernelNames: registeredKernels.map(k => k.kernelName),
      backend,
      predictions: predictions
    }
  });
}

self.addEventListener('message', function(e) {
  try {
    main(e.data.modelUrl);
  } catch (e) {
    self.postMessage({error: true, payload: e});
  }
}, false);
