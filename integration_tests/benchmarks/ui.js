/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import * as tfc from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

export function status(statusText) {
  document.getElementById('status').textContent = statusText;
}

export function setMetadata(metadata) {
  document.getElementById('benchmark-timestamp').textContent = new Date().toISOString();
  document.getElementById('pyKerasVersion').textContent = metadata.keras_version;
  document.getElementById('tfVersion').textContent = metadata.tensorflow_version;
  document.getElementById('tfUsesGPU').textContent = metadata.tensorflow_uses_gpu;
  document.getElementById('tfjs-core-version').textContent = tfc.version_core;
  document.getElementById('tfjs-layers-version').textContent = tfl.version_layers;
}

export function setRunBenchmarksFunction(runAllBenchmarks) {
  const runButton = document.getElementById('runBenchmarks');
  runButton.addEventListener('click', async () => runAllBenchmarks());
}

export function addResult(modelName, result) {
  let row = '<td>' + modelName + '</td>';
  row += '<td>' + result.originalData.description + '</td>';
  row += '<td>' + result.originalData.batch_size + '</td>';
  row += '<td>' + result.originalData.train_epochs + '</td>';
  row += '<td>' + (result.originalData.train_time * 1e3).toFixed(1) + '</td>';
  row += '<td>' + result.trainTimeMs.toFixed(1) + '</td>';
  row += '<td>' + (result.originalData.predict_time * 1e3).toFixed(1) + '</td>';
  row += '<td>' + result.predictTimeMs.toFixed(1) + '</td>';

  document.getElementById('resultsBody').innerHTML += '<tr>' + row + '</tr>';
}
