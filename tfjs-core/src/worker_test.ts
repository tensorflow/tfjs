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

import '@tensorflow/tfjs-backend-cpu';
import {expectArraysClose} from './test_util';

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

let a = tf.tensor1d([1, 2, 3]);
const b = tf.tensor1d([3, 2, 1]);
a = tf.add(a, b);
self.postMessage({data: a.dataSync()});
`;

describe('computation in worker', () => {
  it('tensor in worker', (done) => {
    const worker = new Worker(str2workerURL(workerTest));
    worker.onmessage = (msg) => {
      const data = msg.data.data;
      expectArraysClose(data, [4, 4, 4]);
      done();
    };
  });
});
