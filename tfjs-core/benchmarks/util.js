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

async function getPredictionData(prediction) {
  let output = prediction;
  if (output instanceof Promise) {
    output = await output;
  }
  if (output instanceof tf.Tensor) {
    output = await output.data();
  }
  return output;
}

function arraysClose(n1, n2) {
  const epsilon = 1e-3;

  if (n1 === n2) {
    return true;
  }
  if (n1 == null || n2 == null) {
    return false;
  }

  if (n1.length !== n2.length) {
    return false;
  }
  for (let i = 0; i < n1.length; i++) {
    if (Math.abs(n1[i] - n2[i]) > epsilon) {
      return false;
    }
  }
  return true;
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
