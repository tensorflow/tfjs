/**
 * @license
 * Copyright 2022 Google LLC.
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

import '@tensorflow/tfjs-backend-cpu';

const str2workerURL = (str: string): string => {
  const blob =
      new Blob([str], {type: 'application/javascript'});
  return URL.createObjectURL(blob);
};

// The source code of a web worker.
const workerTest = `
importScripts(location.origin + '/base/tfjs/tfjs-core/tf-core.min.js');
importScripts(location.origin
  + '/base/tfjs/tfjs-backend-cpu/tf-backend-cpu.min.js');
// Import order matters. TFLite must be imported after tfjs core.
importScripts(location.origin + '/base/tfjs/tfjs-tflite/tf-tflite.min.js');

// Setting wasm path is required. It can be set to CDN if needed,
// but that's not a good idea for a test.
tflite.setWasmPath('/base/tfjs/tfjs-tflite/wasm/');
async function main() {
  // This is a test model that adds two tensors of shape [1, 4].
  const model = await tflite.loadTFLiteModel(location.origin + '/base/tfjs/tfjs-tflite/test_files/add4.tflite');

  const a = tf.tensor2d([[1, 2, 3, 4]]);
  const b = tf.tensor2d([[5, 6, 7, 8]]);
  const output = model.predict([a, b]);

  self.postMessage({data: output.dataSync()});
}

main();
`;

describe('tflite in worker', () => {
  it('runs a model', (done) => {
    const worker = new Worker(str2workerURL(workerTest));
    worker.onmessage = (msg) => {
      const data = msg.data.data;
      expect([...data]).toEqual([6, 8, 10, 12]);
      done();
    };
  }, 15_000);
});
