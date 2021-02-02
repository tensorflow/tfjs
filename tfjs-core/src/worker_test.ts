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

import * as tf from './index';
import {describeWithFlags, HAS_WORKER} from './jasmine_util';
import {expectArraysClose} from './test_util';

const fn2workerURL = (fn: Function): string => {
  const blob =
      new Blob(['(' + fn.toString() + ')()'], {type: 'application/javascript'});
  return URL.createObjectURL(blob);
};

// The source code of a web worker.
const workerTest = () => {
  //@ts-ignore
  importScripts('http://bs-local.com:12345/base/dist/tf-core.min.js');
  //@ts-ignore
  importScripts('http://bs-local.com:12345/base/dist/tf-backend-cpu.min.js');
  let a = tf.tensor1d([1, 2, 3]);
  const b = tf.tensor1d([3, 2, 1]);
  a = tf.add(a, b);
  //@ts-ignore
  self.postMessage({data: a.dataSync()});
};

describeWithFlags('computation in worker', HAS_WORKER, () => {
  it('tensor in worker', (done) => {
    const worker = new Worker(fn2workerURL(workerTest));
    worker.onmessage = (msg) => {
      const data = msg.data.data;
      expectArraysClose(data, [4, 4, 4]);
      done();
    };
  });
});
