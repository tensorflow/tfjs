import {setWasmPath} from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

import wasmPath from '../node_modules/@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm';

async function run() {
  setWasmPath(wasmPath);
  await tf.setBackend('wasm');
  tf.add(5, 3).print();
}

run();
