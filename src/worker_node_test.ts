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

import {HAS_NODE_WORKER, describeWithFlags} from './jasmine_util';
import {expectArraysClose} from './test_util';

const fn2String = (fn: Function): string => {
  const funcStr = '('+fn.toString()+')()';
  return funcStr;
};

// The source code of a web worker.
const workerTestNode = () => {
  const tf = require(`${process.cwd()}/dist/tf-core.js`);
  // tslint:disable-next-line:no-require-imports
  const {parentPort} = require('worker_threads');
  let a = tf.tensor1d([1, 2, 3]);
  const b = tf.tensor1d([3, 2, 1]);
  a = a.add(b);
  parentPort.postMessage({data: a.dataSync()});
};

describeWithFlags('computation in worker (node env)', HAS_NODE_WORKER, () => {
  it('tensor in worker', (done) => {
    // tslint:disable-next-line:no-require-imports
    const {Worker} = require('worker_threads');
    const worker = new Worker(fn2String(workerTestNode), {eval: true});
    // tslint:disable-next-line:no-any
    worker.on('message', (msg: any) => {
      const data = msg.data;
      expectArraysClose(data, [4, 4, 4]);
      done();
    });
  });
});
