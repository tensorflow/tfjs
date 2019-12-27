/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

Error.stackTraceLimit = Infinity;

import * as tf from '@tensorflow/tfjs-node';

const a = tf.tensor2d([1, 2, 3, 4], [2, 2], 'float32');
const b = tf.tensor2d([5, 6, 7, 8], [2, 2], 'float32');
const c = a.matMul(b);
console.log(c.dataSync());

async function loadModel() {
  const model = await tf.node.loadSavedModel(
      '../../test_objects/times_three_float', ['serve'], 'serving_default');
  model.dispose();
}
loadModel();
