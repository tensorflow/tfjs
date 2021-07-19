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

import '@tensorflow/tfjs-core/dist/public/chained_ops/to_float';
import '@tensorflow/tfjs-core/dist/public/chained_ops/expand_dims';
import '@tensorflow/tfjs-core/dist/public/chained_ops/resize_bilinear';
import '@tensorflow/tfjs-core/dist/public/chained_ops/squeeze';
import '@tensorflow/tfjs-core/dist/public/chained_ops/reshape';
import '@tensorflow/tfjs-core/dist/public/chained_ops/div';

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs';
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm/dist/backend_wasm';

import wasmSimdPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm';
import wasmSimdThreadedPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm';
import wasmPath from './node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPath,
  'tfjs-backend-wasm-simd.wasm': wasmSimdPath,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmSimdThreadedPath
});

import {STUBBED_IMAGE_VALS} from './test_data';

async function main() {
  tf.setBackend('wasm');
  tf.env().set('WASM_HAS_SIMD_SUPPORT', false);
  tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', false);
  self.postMessage({msg: true, payload: 'in worker main function'});
  await tf.ready();

  const backend = tf.getBackend();
  self.postMessage({msg: true, payload: `'${backend}' backend ready`});

  const registeredKernels = tf.getKernelsForBackend(backend)
  // Debug messsage with info about the registered kernels.
  self.postMessage({
    msg: true,
    payload: {
      numKernels: registeredKernels.length,
      kernelNames: registeredKernels.map(k => k.kernelName),
      backend,
    }
  });

  const model = await blazeface.load();

  self.postMessage({msg: true, payload: `model loaded`});

  let predictions;
  try {
    const input = tf.tensor3d(STUBBED_IMAGE_VALS, [128, 128, 3]);
    predictions = await model.estimateFaces(input, false /*returnTensors*/);
    input.dispose();
  } catch (e) {
    self.postMessage({error: true, payload: {e}});
  }

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
    main();
  } catch (e) {
    self.postMessage({error: true, payload: e});
  }
}, false);
