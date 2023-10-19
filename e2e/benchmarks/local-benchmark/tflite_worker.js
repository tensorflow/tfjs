/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
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

// Load scripts from jsdelivr because it correctly sets the
// "cross-origin-resource-policy" header.
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core/dist/tf-core.js');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu/dist/tf-backend-cpu.js');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.js');
importScripts('https://cdn.jsdelivr.net/npm/comlink@latest/dist/umd/comlink.js');

tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');

const tfliteWorkerAPI = {
  async loadTFLiteModel(modelPath, options) {
    const model = await tflite.loadTFLiteModel(modelPath, options);

    const wrapped = {
      inputs: model.inputs,
      getProfilingResults() {
        return model.getProfilingResults();
      },
      cleanUp() {
        model.modelRunner.cleanUp();
      },
      async predict(inputData) {
        const inputTensorArray = [];
        let outputTensor;
        if (!inputData[0].length) {
          // Single input, move it into an arrary
          inputData = [inputData];
        }
        for (let i = 0; i< this.inputs.length; i++) {
          const inputTensor = tf.tensor(
              inputData[i], this.inputs[i].shape, this.inputs[i].dtype);
          inputTensorArray.push(inputTensor);
        }
        outputTensor = model.predict(inputTensorArray);
        if (model.outputs.length > 1) {
          // Multiple outputs
          let outputData = [];
          for (let tensorName in outputTensor) {
            outputData.push(outputTensor[tensorName].dataSync());
          }
        } else {
          // Single output
          const outputData = outputTensor.dataSync();
        }
        // dispose input and output tensors
        tf.dispose(inputTensorArray);
        tf.dispose(outputTensor);
        // We encourage the user to process output data in the worker thread
        // rather than posting output data to main thread directly, as
        // this would bring much overhead if the output size is huge.
        // From this perspective, we don't post output data to main thread
        // in this benchmark.
        return 'OK';
      }
    };

    return Comlink.proxy(wrapped);
  },
};

Comlink.expose(tfliteWorkerAPI);
