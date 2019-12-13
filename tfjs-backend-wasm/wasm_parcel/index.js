import '@tensorflow/tfjs-backend-wasm';

import * as tf from '@tensorflow/tfjs-core';

async function run() {
  await tf.setBackend('wasm');
  tf.add(5, 3).print();
}

run();
