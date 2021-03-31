import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';
import {Tensor} from '@tensorflow/tfjs-core';

import {loadTFLiteModel} from './tflite_model';

async function start() {
  const tfliteModel = await loadTFLiteModel('mobilenetv2.tflite');
  const img = tf.browser.fromPixels(document.querySelector('img')!);
  const input =
      tf.sub(tf.div(tf.expandDims(tf.cast(img, 'float32')), 127.5), 1);
  const outputTensors = tfliteModel.predict(input, {}) as Tensor[];
  const ouptutTensor = outputTensors[0];
  const outputValues = Array.from(ouptutTensor.dataSync());
  outputValues.shift();
  const sortedResult = outputValues
                           .map((logit, i) => {
                             return {i, logit};
                           })
                           .sort((a, b) => b.logit - a.logit);
  // Show result.
  const classIndex = sortedResult[0].i;
  const score = sortedResult[0].logit;
  console.log(classIndex, score);
}

start();
