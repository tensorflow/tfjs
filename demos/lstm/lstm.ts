/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
import * as dl from 'deeplearn';

// manifest.json lives in the same directory.
const reader = new dl.CheckpointLoader('.');
reader.getAllVariables().then(async vars => {
  const primerData = 3;
  const expected = [1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4];
  const math = dl.ENV.math;

  const lstmKernel1 =
      vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as dl.Array2D;
  const lstmBias1 =
      vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as dl.Array1D;

  const lstmKernel2 =
      vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as dl.Array2D;
  const lstmBias2 =
      vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as dl.Array1D;

  const fullyConnectedBiases = vars['fully_connected/biases'] as dl.Array1D;
  const fullyConnectedWeights = vars['fully_connected/weights'] as dl.Array2D;

  const results: number[] = [];

  await math.scope(async () => {
    const forgetBias = dl.Scalar.new(1.0);
    const lstm1 = (data: dl.Array2D, c: dl.Array2D, h: dl.Array2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: dl.Array2D, c: dl.Array2D, h: dl.Array2D) =>
        dl.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);

    let c: dl.Array2D[] = [
      dl.zeros([1, lstmBias1.shape[0] / 4]),
      dl.zeros([1, lstmBias2.shape[0] / 4])
    ];
    let h: dl.Array2D[] = [
      dl.zeros([1, lstmBias1.shape[0] / 4]),
      dl.zeros([1, lstmBias2.shape[0] / 4])
    ];

    let input = primerData;
    for (let i = 0; i < expected.length; i++) {
      const onehot = dl.oneHot(dl.Array1D.new([input]), 10);

      const output = dl.multiRNNCell([lstm1, lstm2], onehot, c, h);

      c = output[0];
      h = output[1];

      const outputH = h[1];
      const logits =
          outputH.matMul(fullyConnectedWeights).add(fullyConnectedBiases);

      const result = await dl.argMax(logits).val();
      results.push(result);
      input = result;
    }
  });
  document.getElementById('expected').innerHTML = expected.toString();
  document.getElementById('results').innerHTML = results.toString();
  if (dl.util.arraysEqual(expected, results)) {
    document.getElementById('success').innerHTML = 'Success!';
  } else {
    document.getElementById('success').innerHTML = 'Failure.';
  }
});
