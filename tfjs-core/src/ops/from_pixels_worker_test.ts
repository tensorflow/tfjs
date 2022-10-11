/**
 * @license
 * Copyright 2021 Google Inc. All Rights Reserved.
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
import {expectArraysEqual} from '../test_util';

describe('fromPixels worker', () => {
  it('fromPixels for ImageBitmap, worker', (done) => {
    // Necessary preconditions
    if (typeof (createImageBitmap) === 'undefined' ||
        typeof (Worker) === 'undefined') {
      done();
      return;
    }

    // Test-only preconditions
    if (typeof (ImageData) === 'undefined' || typeof (Blob) === 'undefined' ||
        typeof (URL) === 'undefined') {
      pending('Test-only js APIs are not supported in this context');
      done();
      return;
    }

    const str2workerURL = (str: string): string => {
      const blob = new Blob([str], {type: 'application/javascript'});
      return URL.createObjectURL(blob);
    };

    // The source code of a web worker.
    const workerTest = `
    importScripts(location.origin + '/base/tfjs/tfjs-core/tf-core.min.js');
    importScripts(location.origin
      + '/base/tfjs/tfjs-backend-cpu/tf-backend-cpu.min.js');

    self.onmessage = (msg) => {
      const bitmap = msg.data;
      const tensor = tf.browser.fromPixels(bitmap, 4);
      tensor.data().then((data) => {
        const thinTensor = {shape: tensor.shape, dtype: tensor.dtype, data: data};
        self.postMessage(thinTensor);
      });
    };
    `;

    const worker = new Worker(str2workerURL(workerTest));

    worker.onmessage = (msg) => {
      const thinTensor = msg.data;
      expect(thinTensor.shape).toEqual([1, 1, 4]);
      expect(thinTensor.dtype).toBe('int32');
      expectArraysEqual(thinTensor.data, [1, 2, 3, 255]);
      done();
      worker.terminate();
    };

    worker.onerror = (e) => {
      if(typeof OffscreenCanvas === 'undefined' ||
         typeof OffscreenCanvasRenderingContext2D === 'undefined') {
        expect(e.message).toEqual(
            'Cannot parse input in current context. ' +
            'Reason: OffscreenCanvas Context2D rendering is not supported.'
        );
      } else {
        throw e;
      }
      done();
      worker.terminate();
    };

    const imData = new ImageData(new Uint8ClampedArray([1, 2, 3, 255]), 1, 1);
    createImageBitmap(imData).then((bitmap) => {
      worker.postMessage(bitmap, [bitmap]);
    });
  });
});
