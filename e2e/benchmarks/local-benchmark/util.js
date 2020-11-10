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

async function convertTensorToData(tensor) {
  const data = await tensor.data();
  tensor.dispose();
  return data;
}

async function getPredictionData(output) {
  if (output instanceof Promise) {
    output = await output;
  }

  if (output instanceof tf.Tensor) {
    output = await convertTensorToData(output);
  } else if (Array.isArray(output)) {
    for (let i = 0; i < output.length; i++) {
      if (output[i] instanceof tf.Tensor) {
        output[i] = await convertTensorToData(output[i]);
      }
    }
  } else if (output != null && typeof output === 'object') {
    for (const property in output) {
      if (output[property] instanceof tf.Tensor) {
        output[property] = await convertTensorToData(output[property]);
      }
    }
  }
  return output;
}

function printTime(elapsed) {
  return elapsed.toFixed(1) + ' ms';
}

function printMemory(bytes) {
  if (bytes < 1024) {
    return bytes + ' B';
  } else if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(2) + ' KB';
  } else {
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }
}

function sleep(timeMs) {
  return new Promise(resolve => setTimeout(resolve, timeMs));
}

function queryTimerIsEnabled() {
  return _tfengine.ENV.getNumber(
             'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0;
}
