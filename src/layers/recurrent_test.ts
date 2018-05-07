/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit tests for recurrent.ts.
 */

// tslint:disable:max-line-length
import {Scalar, scalar, Tensor, tensor2d, tensor3d, train} from '@tensorflow/tfjs-core';

import * as K from '../backend/tfjs_backend';
import * as tfl from '../index';
import * as metrics from '../metrics';
import {ModelAndWeightsConfig, modelFromJSON} from '../models';
import {DType, Kwargs} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {RNN, RNNCell} from './recurrent';
// tslint:enable:max-line-length

/**
 * A simplistic RNNCell for testing.
 *
 * This RNNCell performs the following with the inputs and states.
 * - calculates a reduced mean over all input elements,
 * - adds that mean to the state tensor(s),
 * - take the negative of the 1st current state tensor and use it as the
 *   output.
 */
class RNNCellForTest extends RNNCell {
  static className = 'RNNCellForTest';
  constructor(stateSizes: number|number[]) {
    super({});
    this.stateSize = stateSizes;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = inputs as Tensor[];
    const dataInputs = inputs[0];
    const states = inputs.slice(1);
    const mean = K.mean(dataInputs) as Scalar;
    const newStates = states.map(state => K.scalarPlusArray(mean, state));
    const output = K.neg(newStates[0]);
    return [output].concat(newStates);
  }
}

describeMathCPU('RNN-Layer', () => {
  // TODO(cais): Add tests for stacked RNN cell (i.e., multiple cells) once it
  //   implemented.
  // TODO(cais): Add tests for stateful RNN once it is implemented.
  // TODO(cais): Add tests for masks once implemented.
  // TODO(cais): Add tests for constants once implemented.

  it('constructor: only cell', () => {
    const cell = new RNNCellForTest(5);
    const rnn = tfl.layers.rnn({cell}) as RNN;
    expect(rnn.returnSequences).toEqual(false);
    expect(rnn.returnState).toEqual(false);
    expect(rnn.goBackwards).toEqual(false);
  });

  it('constructor: cell and custom options', () => {
    const cell = new RNNCellForTest(5);
    const rnn = tfl.layers.rnn({
      cell,
      returnSequences: true,
      returnState: true,
      goBackwards: true
    }) as RNN;
    expect(rnn.returnSequences).toEqual(true);
    expect(rnn.returnState).toEqual(true);
    expect(rnn.goBackwards).toEqual(true);
  });

  it('computeOutputShape: 1 state, returnSequences=false, returnState=false',
     () => {
       const cell = new RNNCellForTest(5);
       const rnn = tfl.layers.rnn({cell});
       const inputShape = [4, 3, 2];
       expect(rnn.computeOutputShape(inputShape)).toEqual([4, 5]);
     });

  it('computeOutputShape: 1 state, returnSequences=true, returnState=false',
     () => {
       const cell = new RNNCellForTest([5, 6]);
       const rnn = tfl.layers.rnn({cell, returnSequences: true});
       const inputShape = [4, 3, 2];
       expect(rnn.computeOutputShape(inputShape)).toEqual([4, 3, 5]);
     });

  it('computeOutputShape: 1 state, returnSequences=true, returnState=true',
     () => {
       const cell = new RNNCellForTest(6);
       const rnn =
           tfl.layers.rnn({cell, returnSequences: true, returnState: true});
       const inputShape = [4, 3, 2];
       expect(rnn.computeOutputShape(inputShape)).toEqual([[4, 3, 6], [4, 6]]);
     });

  it('computeOutputShape: 2 states, returnSequences=true, returnState=true',
     () => {
       const cell = new RNNCellForTest([5, 6]);
       const rnn =
           tfl.layers.rnn({cell, returnSequences: true, returnState: true});
       const inputShape = [4, 3, 2];
       expect(rnn.computeOutputShape(inputShape)).toEqual([
         [4, 3, 5], [4, 5], [4, 6]
       ]);
     });

  it('apply: Symbolic: 1 state, returnSequences=false, returnState=false',
     () => {
       const cell = new RNNCellForTest(6);
       const rnn = tfl.layers.rnn({cell});
       const input =
           new tfl.SymbolicTensor(DType.float32, [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor;
       expect(output.shape).toEqual([16, 6]);
     });

  it('apply: Symbolic: 1 state, returnSequences=true, returnState=false',
     () => {
       const cell = new RNNCellForTest(6);
       const rnn = tfl.layers.rnn({cell, returnSequences: true});
       const input =
           new tfl.SymbolicTensor(DType.float32, [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor;
       expect(output.shape).toEqual([16, 10, 6]);
     });

  it('apply: Symbolic: 1 state, returnSequences=true, returnState=true', () => {
    const cell = new RNNCellForTest(6);
    const rnn =
        tfl.layers.rnn({cell, returnSequences: true, returnState: true});
    const input =
        new tfl.SymbolicTensor(DType.float32, [16, 10, 8], null, [], null);
    const output = rnn.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([16, 10, 6]);
    expect(output[1].shape).toEqual([16, 6]);
  });

  it('apply: Symbolic: 1 state, returnSequences=false, returnState=true',
     () => {
       const cell = new RNNCellForTest(6);
       const rnn =
           tfl.layers.rnn({cell, returnSequences: false, returnState: true});
       const input =
           new tfl.SymbolicTensor(DType.float32, [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor[];
       expect(output.length).toEqual(2);
       expect(output[0].shape).toEqual([16, 6]);
       expect(output[1].shape).toEqual([16, 6]);
     });

  it('apply: Symbolic: 2 states, returnSequences=true, returnState=true',
     () => {
       const cell = new RNNCellForTest([5, 6]);
       const rnn =
           tfl.layers.rnn({cell, returnSequences: true, returnState: true});
       const input =
           new tfl.SymbolicTensor(DType.float32, [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor[];
       expect(output.length).toEqual(3);
       expect(output[0].shape).toEqual([16, 10, 5]);
       expect(output[1].shape).toEqual([16, 5]);
       expect(output[2].shape).toEqual([16, 6]);
     });
});

describeMathCPUAndGPU('RNN-Layer-Math', () => {
  it('getInitialState: 1 state', () => {
    const cell = new RNNCellForTest(5);
    const inputs = K.zeros([4, 3, 2]);
    const rnn = tfl.layers.rnn({cell}) as RNN;
    const initialStates = rnn.getInitialState(inputs);
    expect(initialStates.length).toEqual(1);
    expectTensorsClose(initialStates[0], K.zeros([4, 5]));
  });

  it('getInitialState: 2 states', () => {
    const cell = new RNNCellForTest([5, 6]);
    const inputs = K.zeros([4, 3, 2]);
    const rnn = tfl.layers.rnn({cell}) as RNN;
    const initialStates = rnn.getInitialState(inputs);
    expect(initialStates.length).toEqual(2);
    expectTensorsClose(initialStates[0], K.zeros([4, 5]));
    expectTensorsClose(initialStates[1], K.zeros([4, 6]));
  });

  it('call: 1 state: returnSequences=false, returnState=false', () => {
    const cell = new RNNCellForTest(4);
    const rnn = tfl.layers.rnn({cell});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs) as Tensor;
    expectTensorsClose(
        outputs, K.scalarTimesArray(scalar(-57.75), K.ones([2, 4])));
  });

  it('apply: 1 state: returnSequences=true, returnState=false', () => {
    const cell = new RNNCellForTest(3);
    const rnn = tfl.layers.rnn({cell, returnSequences: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs) as Tensor;
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
            ],
            [2, 3, 3]));
  });

  it('apply: 1 state: returnSequences=true, returnState=true', () => {
    const cell = new RNNCellForTest(3);
    const rnn =
        tfl.layers.rnn({cell, returnSequences: true, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs) as Tensor[];
    expect(outputs.length).toEqual(2);
    expectTensorsClose(
        outputs[0],
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
            ],
            [2, 3, 3]));
    expectTensorsClose(
        outputs[1],
        tensor2d([[57.75, 57.75, 57.75], [57.75, 57.75, 57.75]], [2, 3]));
  });

  it('apply: 2 states: returnSequences=true, returnState=true', () => {
    const cell = new RNNCellForTest([3, 4]);
    const rnn =
        tfl.layers.rnn({cell, returnSequences: true, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs) as Tensor[];
    expect(outputs.length).toEqual(3);
    expectTensorsClose(
        outputs[0],
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25], [-27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75]
              ],
            ],
            [2, 3, 3]));
    expectTensorsClose(
        outputs[1],
        tensor2d([[57.75, 57.75, 57.75], [57.75, 57.75, 57.75]], [2, 3]));
    expectTensorsClose(
        outputs[2],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
  });

  it('call: with 1 initialState', () => {
    const cell = new RNNCellForTest(4);
    const rnn = tfl.layers.rnn({cell});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs =
        rnn.apply(inputs, {'initialState': [K.ones([2, 4])]}) as Tensor;
    expectTensorsClose(
        outputs, K.scalarTimesArray(scalar(-58.75), K.ones([2, 4])));
  });

  it('call: with 2 initialStates', () => {
    const cell = new RNNCellForTest([4, 5]);
    const rnn = tfl.layers.rnn({cell, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs, {
      'initialState':
          [K.ones([2, 4]), K.scalarTimesArray(scalar(2), K.ones([2, 5]))]
    }) as Tensor[];
    expect(outputs.length).toEqual(3);
    expectTensorsClose(
        outputs[0], K.scalarTimesArray(scalar(-58.75), K.ones([2, 4])));
    expectTensorsClose(
        outputs[1], K.scalarTimesArray(scalar(58.75), K.ones([2, 4])));
    expectTensorsClose(
        outputs[2], K.scalarTimesArray(scalar(59.75), K.ones([2, 5])));
  });

  it('call with incorrect number of initialStates leads to ValueError', () => {
    const cell = new RNNCellForTest([4, 5]);
    const rnn = tfl.layers.rnn({cell, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    expect(() => rnn.apply(inputs, {
      'initialState': [K.ones([2, 4])]
    })).toThrowError(/An initialState was passed that is not compatible with/);
  });
});

describeMathCPU('SimpleRNN Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5, returnState: true});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5, returnSequences: true});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({
      units: 5,
      returnSequences: true,
      returnState: true,
    });
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 10, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });
});

describeMathCPUAndGPU('SimpleRNN Tensor', () => {
  const units = 5;
  const batchSize = 4;
  const inputSize = 2;

  // TODO(cais): Add test for the default recurrent initializer ('Orthogonal')
  //   when it becomes available.
  // TODO(cais): Test dropout and recurrentDropout when implemented.

  const activations = ['linear', 'tanh'];
  for (const activation of activations) {
    const testTitle =
        `returnSequences=false, returnState=false, useBias=true, ${activation}`;
    it(testTitle, () => {
      const timeSteps = 1;
      const simpleRNN = tfl.layers.simpleRNN({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        activation
      });
      const input = K.ones([batchSize, timeSteps, inputSize]);
      const output = simpleRNN.apply(input) as Tensor;
      let expectedElementValue = inputSize + 1;
      if (activation === 'tanh') {
        expectedElementValue = Math.tanh(expectedElementValue);
      }
      expectTensorsClose(
          output,
          K.scalarTimesArray(
              scalar(expectedElementValue), K.ones([batchSize, units])));
    });
  }

  const returnStateValues = [false, true];
  for (const returnState of returnStateValues) {
    const testTitle = `returnSequences=true, ` +
        `returnState=${returnState}, useBias=true, linear`;
    it(testTitle, () => {
      const timeSteps = 2;
      const simpleRNN = tfl.layers.simpleRNN({
        units,
        returnSequences: true,
        returnState,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        activation: 'linear'
      });
      const input = K.ones([batchSize, timeSteps, inputSize]);
      let output = simpleRNN.apply(input);
      let finalState: Tensor;
      if (returnState) {
        output = output as Tensor[];
        expect(output.length).toEqual(2);
        finalState = output[1];
        output = output[0];
      } else {
        output = output as Tensor;
      }

      expect(output.shape).toEqual([batchSize, timeSteps, units]);
      const timeMajorOutput = K.transpose(output, [1, 0, 2]);
      const outputT0 = K.sliceAlongFirstAxis(timeMajorOutput, 0, 1);
      const outputT1 = K.sliceAlongFirstAxis(timeMajorOutput, 1, 1);
      expectTensorsClose(
          outputT0,
          K.scalarTimesArray(
              scalar(inputSize + 1), K.ones([1, batchSize, units])));
      expectTensorsClose(
          outputT1,
          K.scalarTimesArray(
              scalar((inputSize + 1) * (units + 1)),
              K.ones([1, batchSize, units])));
      if (returnState) {
        expectTensorsClose(finalState, outputT1.reshape([batchSize, units]));
      }
    });
  }

  it('BPTT', () => {
    // The following golden values for assertion can be obtained with the
    // following Python Keras code.
    // ```python
    // import keras
    // import numpy as np
    //
    // sequence_length = 3
    // input_size = 4
    // batch_size = 5
    //
    // t_input = keras.Input([sequence_length, input_size])
    // simple_rnn = keras.layers.SimpleRNN(1,
    //                                     kernel_initializer='ones',
    //                                     recurrent_initializer='ones',
    //                                     use_bias=False)
    // dense = keras.layers.Dense(1,
    //                            kernel_initializer='ones',
    //                            use_bias=False)
    // output = dense(simple_rnn(t_input))
    // model = keras.Model(t_input, output)
    // optimizer = keras.optimizers.SGD(1)
    // model.compile(optimizer=optimizer, loss='mean_squared_error')
    //
    // x = np.ones([batch_size, sequence_length, input_size])
    // y = np.zeros([batch_size, 1])
    // model.fit(x, y, batch_size=batch_size, epochs=2)
    // print(simple_rnn.get_weights()[0])
    // print(simple_rnn.get_weights()[1])
    // print(dense.get_weights()[0])
    // ```
    const sequenceLength = 3;
    const inputSize = 4;
    const batchSize = 5;
    const simpleRNN = tfl.layers.simpleRNN({
      units: 1,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      useBias: false,
    });
    const dense = tfl.layers.dense({
      units: 1,
      kernelInitializer: 'ones',
      useBias: false,
    });

    const sgd = train.sgd(5);
    const x = K.ones([batchSize, sequenceLength, inputSize]);
    const y = K.zeros([batchSize, 1]);
    dense.apply(simpleRNN.apply(x));
    const lossFn = () => {
      return K.mean(metrics.mse(y, dense.apply(simpleRNN.apply(x)) as Tensor))
          .asScalar();
    };
    for (let i = 0; i < 2; ++i) {
      sgd.minimize(lossFn);
    }
    expectTensorsClose(
        simpleRNN.getWeights()[0],
        K.scalarTimesArray(scalar(0.8484658), K.ones([4, 1])));
    expectTensorsClose(
        simpleRNN.getWeights()[1],
        K.scalarTimesArray(scalar(0.8484799), K.ones([1, 1])));
    expectTensorsClose(
        dense.getWeights()[0],
        K.scalarTimesArray(scalar(80.967026), K.ones([1, 1])));
  });
});

describeMathCPU('GRU Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5});
    const output = gru.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5, returnState: true});
    const output = gru.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5, returnSequences: true});
    const output = gru.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({
      units: 5,
      returnSequences: true,
      returnState: true,
    });
    const output = gru.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 10, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });

  it('trainableWeights, nonTrainableWeights and weights give correct outputs',
     () => {
       const input =
           new tfl.SymbolicTensor(DType.float32, [2, 3, 4], null, [], null);
       const gru = tfl.layers.gru({units: 5, returnState: true});
       gru.apply(input);
       expect(gru.trainable).toEqual(true);
       // Trainable weights: kernel, recurrent kernel and bias.
       expect(gru.trainableWeights.length).toEqual(3);
       expect(gru.nonTrainableWeights.length).toEqual(0);
       expect(gru.weights.length).toEqual(3);
     });
});

describeMathCPUAndGPU('GRU Tensor', () => {
  // Note:
  // The golden expected values used for assertions in these unit tests can be
  // obtained through running the Python code similar to the following example.
  // TensorFlow 1.5 was used to obtain the values.
  //
  // ```python
  // import numpy as np
  // import tensorflow as tf
  //
  // units = 5
  // batch_size = 4
  // input_size = 2
  // time_steps = 3
  //
  // with tf.Session() as sess:
  //   lstm = tf.keras.layers.GRU(units,
  //                              kernel_initializer="ones",
  //                              recurrent_initializer="ones",
  //                              bias_initializer="ones",
  //                              return_sequences=True,
  //                              return_state=True)
  //   inputs = tf.placeholder(tf.float32, shape=[None, None, input_size])
  //   outputs = lstm(inputs)
  //
  //   sess.run(tf.global_variables_initializer())
  //   feed_inputs = np.ones([batch_size, time_steps, input_size])
  //   print(sess.run(outputs, feed_dict={inputs: feed_inputs}))
  // ```

  const units = 5;
  const batchSize = 4;
  const inputSize = 2;
  const timeSteps = 3;
  const goldenOutputElementValues = [0.22847827, 0.2813754, 0.29444352];

  // TODO(cais): Add test for the default recurrent initializer ('Orthogonal')
  //   when it becomes available.
  // TODO(cais): Test dropout and recurrentDropout when implemented.

  const implementations = [1, 2];
  const returnStateValues = [false, true];
  const returnSequencesValues = [false, true];
  for (const implementation of implementations) {
    for (const returnState of returnStateValues) {
      for (const returnSequences of returnSequencesValues) {
        const testTitle = `implementation=${implementation}, ` +
            `returnSequences=${returnSequences}, ` +
            `returnState=${returnState}`;
        it(testTitle, () => {
          const gru = tfl.layers.gru({
            units,
            kernelInitializer: 'ones',
            recurrentInitializer: 'ones',
            biasInitializer: 'ones',
            returnState,
            returnSequences,
            implementation
          });
          const input = K.zeros([batchSize, timeSteps, inputSize]);
          let output = gru.apply(input);

          const goldenOutputElementValueFinal =
              goldenOutputElementValues[goldenOutputElementValues.length - 1];

          let expectedOutput: Tensor;
          if (returnSequences) {
            const outputs = goldenOutputElementValues.map(
                value => K.scalarTimesArray(
                    scalar(value), K.ones([1, batchSize, units])));
            expectedOutput = K.transpose(
                K.concatAlongFirstAxis(
                    K.concatAlongFirstAxis(outputs[0], outputs[1]), outputs[2]),
                [1, 0, 2]);
          } else {
            expectedOutput = K.scalarTimesArray(
                scalar(goldenOutputElementValueFinal),
                K.ones([batchSize, units]));
          }
          if (returnState) {
            output = output as Tensor[];
            expect(output.length).toEqual(2);
            expectTensorsClose(output[0], expectedOutput);
            expectTensorsClose(
                output[1],
                K.scalarTimesArray(
                    scalar(goldenOutputElementValueFinal),
                    K.ones([batchSize, units])));
          } else {
            output = output as Tensor;
            expectTensorsClose(output, expectedOutput);
          }
        });
      }
    }
  }

  it('BPTT', () => {
    // The following golden values for assertion can be obtained with the
    // following Python Keras code.
    // ```python
    // import keras
    // import numpy as np

    // sequence_length = 3
    // input_size = 4
    // batch_size = 5

    // t_input = keras.Input([sequence_length, input_size])
    // gru = keras.layers.GRU(1,
    //                        kernel_initializer='zeros',
    //                        recurrent_initializer='zeros',
    //                        use_bias=False)
    // dense = keras.layers.Dense(1,
    //                            kernel_initializer='ones',
    //                            use_bias=False)
    // output = dense(gru(t_input))
    // model = keras.Model(t_input, output)
    // optimizer = keras.optimizers.SGD(1)
    // model.compile(optimizer=optimizer, loss='mean_squared_error')

    // x = np.ones([batch_size, sequence_length, input_size])
    // y = np.ones([batch_size, 1])
    // model.fit(x, y, batch_size=batch_size, epochs=2)
    // print(gru.get_weights()[0])
    // print(gru.get_weights()[1])
    // print(dense.get_weights()[0])
    // ```
    const sequenceLength = 3;
    const inputSize = 4;
    const batchSize = 5;
    const gru = tfl.layers.gru({
      units: 1,
      kernelInitializer: 'zeros',
      recurrentInitializer: 'zeros',
      useBias: false
    });
    const dense =
        tfl.layers.dense({units: 1, kernelInitializer: 'ones', useBias: false});

    const sgd = train.sgd(1);
    const x = K.ones([batchSize, sequenceLength, inputSize]);
    const y = K.ones([batchSize, 1]);
    dense.apply(gru.apply(x));
    const lossFn = () => {
      return K.mean(metrics.mse(y, dense.apply(gru.apply(x)) as Tensor))
          .asScalar();
    };
    for (let i = 0; i < 2; ++i) {
      sgd.minimize(lossFn);
    }
    expectTensorsClose(
        gru.getWeights()[0],
        K.tile(tensor2d([[-0.03750037, 0, 1.7500007]], [1, 3]), [4, 1]));
    expectTensorsClose(
        gru.getWeights()[1],
        tensor2d([[-1.562513e-02, 0, 2.086183e-07]], [1, 3]));
    expectTensorsClose(dense.getWeights()[0], tensor2d([[1.2187521]], [1, 1]));
  });
});

describeMathCPU('LSTM Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5});
    const output = lstm.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5, returnState: true});
    const output = lstm.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(3);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
    expect(output[2].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5, returnSequences: true});
    const output = lstm.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input =
        new tfl.SymbolicTensor(DType.float32, [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({
      units: 5,
      returnSequences: true,
      returnState: true,
    });
    const output = lstm.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(3);
    expect(output[0].shape).toEqual([9, 10, 5]);
    expect(output[1].shape).toEqual([9, 5]);
    expect(output[2].shape).toEqual([9, 5]);
  });

  it('trainableWeights, nonTrainableWeights and weights give correct outputs',
     () => {
       const input =
           new tfl.SymbolicTensor(DType.float32, [2, 3, 4], null, [], null);
       const lstm = tfl.layers.lstm({units: 5, returnState: true});
       lstm.apply(input);
       expect(lstm.trainable).toEqual(true);
       // Trainable weights: kernel, recurrent kernel and bias.
       expect(lstm.trainableWeights.length).toEqual(3);
       expect(lstm.nonTrainableWeights.length).toEqual(0);
       expect(lstm.weights.length).toEqual(3);
     });
});

describeMathCPUAndGPU('LSTM Tensor', () => {
  // Note:
  // The golden expected values used for assertions in these unit tests can be
  // obtained through running the Python code similar to the following example.
  // TensorFlow 1.5 was used to obtain the values.
  //
  // ```python
  // import numpy as np
  // import tensorflow as tf
  //
  // units = 5
  // batch_size = 4
  // input_size = 2
  // time_steps = 2
  //
  // with tf.Session() as sess:
  //   lstm = tf.keras.layers.LSTM(units,
  //                               kernel_initializer="ones",
  //                               recurrent_initializer="ones",
  //                               bias_initializer="ones",
  //                               return_sequences=True,
  //                               return_state=True,
  //                               unit_forget_bias=True)
  //   inputs = tf.placeholder(tf.float32, shape=[None, None, input_size])
  //   outputs = lstm(inputs)
  //
  //   sess.run(tf.global_variables_initializer())
  //   feed_inputs = np.ones([batch_size, time_steps, input_size])
  //   print(sess.run(outputs, feed_dict={inputs: feed_inputs}))
  // ```

  const units = 5;
  const batchSize = 4;
  const inputSize = 2;
  const timeSteps = 2;

  // TODO(cais): Add test for the default recurrent initializer ('Orthogonal')
  //   when it becomes available.
  // TODO(cais): Test dropout and recurrentDropout when implemented.

  const implementations: Array<(1 | 2)> = [1, 2];
  const returnStateValues = [false, true];
  const returnSequencesValues = [false, true];
  for (const implementation of implementations) {
    for (const returnState of returnStateValues) {
      for (const returnSequences of returnSequencesValues) {
        const testTitle = `implementation=${implementation}, ` +
            `returnSequences=${returnSequences}, ` +
            `returnState=${returnState}`;
        it(testTitle, () => {
          const lstm = tfl.layers.lstm({
            units,
            kernelInitializer: 'ones',
            recurrentInitializer: 'ones',
            biasInitializer: 'ones',
            returnState,
            returnSequences,
            implementation
          });
          const input = K.ones([batchSize, timeSteps, inputSize]);
          let output = lstm.apply(input);

          // See comments at the beginning of this describe() block on how these
          // golden expected values can be obtained.
          const goldenOutputElementValueAtT0 = 0.7595095;
          const goldenOutputElementValueAtT1 = 0.96367633;
          const goldenHStateElementValue = goldenOutputElementValueAtT1;
          const goldenCStateElementValue = 1.99505234;

          let expectedOutput: Tensor;
          if (returnSequences) {
            const outputAtT0 = K.scalarTimesArray(
                scalar(goldenOutputElementValueAtT0),
                K.ones([1, batchSize, units]));
            const outputAtT1 = K.scalarTimesArray(
                scalar(goldenOutputElementValueAtT1),
                K.ones([1, batchSize, units]));
            expectedOutput = K.transpose(
                K.concatAlongFirstAxis(outputAtT0, outputAtT1), [1, 0, 2]);
          } else {
            expectedOutput = K.scalarTimesArray(
                scalar(goldenOutputElementValueAtT1),
                K.ones([batchSize, units]));
          }
          if (returnState) {
            output = output as Tensor[];
            expect(output.length).toEqual(3);
            expectTensorsClose(output[0], expectedOutput);
            expectTensorsClose(
                output[1],
                K.scalarTimesArray(
                    scalar(goldenHStateElementValue),
                    K.ones([batchSize, units])));
            expectTensorsClose(
                output[2],
                K.scalarTimesArray(
                    scalar(goldenCStateElementValue),
                    K.ones([batchSize, units])));
          } else {
            output = output as Tensor;
            expectTensorsClose(output, expectedOutput);
          }
        });
      }
    }

    it('BPTT', () => {
      // The following golden values for assertion can be obtained with the
      // following Python Keras code.
      // ```python
      // import keras
      // import numpy as np
      //
      // sequence_length = 3
      // input_size = 4
      // batch_size = 5
      //
      // t_input = keras.Input([sequence_length, input_size])
      // lstm = keras.layers.LSTM(1,
      //                          kernel_initializer='zeros',
      //                          recurrent_initializer='zeros',
      //                          use_bias=False)
      // dense = keras.layers.Dense(1,
      //                            kernel_initializer='ones',
      //                            use_bias=False)
      // output = dense(lstm(t_input))
      // model = keras.Model(t_input, output)
      // optimizer = keras.optimizers.SGD(1)
      // model.compile(optimizer=optimizer, loss='mean_squared_error')
      //
      // x = np.ones([batch_size, sequence_length, input_size])
      // y = np.ones([batch_size, 1])
      // model.fit(x, y, batch_size=batch_size, epochs=2)
      // print(lstm.get_weights()[0])
      // print(lstm.get_weights()[1])
      // print(dense.get_weights()[0])
      // ```
      const sequenceLength = 3;
      const inputSize = 4;
      const batchSize = 5;
      const lstm = tfl.layers.lstm({
        units: 1,
        kernelInitializer: 'zeros',
        recurrentInitializer: 'zeros',
        useBias: false,
      });
      const dense = tfl.layers.dense({
        units: 1,
        kernelInitializer: 'ones',
        useBias: false,
      });

      const sgd = train.sgd(1);
      const x = K.ones([batchSize, sequenceLength, inputSize]);
      const y = K.ones([batchSize, 1]);
      dense.apply(lstm.apply(x));
      const lossFn = () => {
        return K.mean(metrics.mse(y, dense.apply(lstm.apply(x)) as Tensor))
            .asScalar();
      };
      for (let i = 0; i < 2; ++i) {
        sgd.minimize(lossFn);
      }
      expectTensorsClose(
          lstm.getWeights()[0],
          K.tile(
              tensor2d(
                  [[0.11455188, 0.06545822, 0.8760446, 0.18237013]], [1, 4]),
              [4, 1]));
      expectTensorsClose(
          lstm.getWeights()[1],
          tensor2d([[0.02831176, 0.01934617, 0.00025817, 0.05784169]], [1, 4]));
      expectTensorsClose(
          dense.getWeights()[0], tensor2d([[1.4559253]], [1, 1]));
    });
  }
});

describeMathCPU('LSTM-deserialization', () => {
  it('modelFromConfig', async done => {
    modelFromJSON(fakeLSTMModel)
        .then(model => {
          const encoderInputs = K.zeros([1, 3, 71], DType.float32);
          const decoderInputs = K.zeros([1, 3, 94], DType.float32);
          const outputs =
              model.predict([encoderInputs, decoderInputs]) as Tensor;
          expect(outputs.shape).toEqual([1, 3, 94]);
          done();
        })
        .catch(done.fail);
  });
});

const fakeLSTMModel: ModelAndWeightsConfig = {
  modelTopology: {
    'class_name': 'Model',
    'keras_version': '2.1.2',
    'config': {
      'layers': [
        {
          'class_name': 'InputLayer',
          'config': {
            'dtype': 'float32',
            'batch_input_shape': [null, null, 71],
            'name': 'input_1',
            'sparse': false
          },
          'inbound_nodes': [],
          'name': 'input_1'
        },
        {
          'class_name': 'InputLayer',
          'config': {
            'dtype': 'float32',
            'batch_input_shape': [null, null, 94],
            'name': 'input_2',
            'sparse': false
          },
          'inbound_nodes': [],
          'name': 'input_2'
        },
        {
          'class_name': 'LSTM',
          'config': {
            'recurrent_activation': 'hard_sigmoid',
            'trainable': true,
            'recurrent_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'use_bias': true,
            'bias_regularizer': null,
            'return_state': true,
            'unroll': false,
            'activation': 'tanh',
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'units': 256,
            'unit_forget_bias': true,
            'activity_regularizer': null,
            'recurrent_dropout': 0.0,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'kernel_constraint': null,
            'dropout': 0.0,
            'stateful': false,
            'recurrent_regularizer': null,
            'name': 'lstm_1',
            'bias_constraint': null,
            'go_backwards': false,
            'implementation': 1,
            'kernel_regularizer': null,
            'return_sequences': false,
            'recurrent_constraint': null
          },
          'inbound_nodes': [[['input_1', 0, 0, {}]]],
          'name': 'lstm_1'
        },
        {
          'class_name': 'LSTM',
          'config': {
            'recurrent_activation': 'hard_sigmoid',
            'trainable': true,
            'recurrent_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'use_bias': true,
            'bias_regularizer': null,
            'return_state': true,
            'unroll': false,
            'activation': 'tanh',
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'units': 256,
            'unit_forget_bias': true,
            'activity_regularizer': null,
            'recurrent_dropout': 0.0,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'kernel_constraint': null,
            'dropout': 0.0,
            'stateful': false,
            'recurrent_regularizer': null,
            'name': 'lstm_2',
            'bias_constraint': null,
            'go_backwards': false,
            'implementation': 1,
            'kernel_regularizer': null,
            'return_sequences': true,
            'recurrent_constraint': null
          },
          'inbound_nodes': [[
            ['input_2', 0, 0, {}], ['lstm_1', 0, 1, {}], ['lstm_1', 0, 2, {}]
          ]],
          'name': 'lstm_2'
        },
        {
          'class_name': 'Dense',
          'config': {
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'name': 'dense_1',
            'kernel_constraint': null,
            'bias_regularizer': null,
            'bias_constraint': null,
            'activation': 'softmax',
            'trainable': true,
            'kernel_regularizer': null,
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'units': 94,
            'use_bias': true,
            'activity_regularizer': null
          },
          'inbound_nodes': [[['lstm_2', 0, 0, {}]]],
          'name': 'dense_1'
        }
      ],
      'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
      'output_layers': [['dense_1', 0, 0]],
      'name': 'model_1'
    },
    'backend': 'tensorflow'
  }
};

describeMathCPU('StackedRNNCells Symbolic', () => {
  it('With SimpleRNNCell', () => {
    const stackedRNN = tfl.layers.rnn({
      cell: tfl.layers.stackedRNNCells({
        cells: [
          tfl.layers.simpleRNNCell(
            { units: 3, recurrentInitializer: 'glorotNormal' }),
          tfl.layers.simpleRNNCell(
            { units: 2, recurrentInitializer: 'glorotNormal' })
        ],
      })
    });
    const input =
        new tfl.SymbolicTensor(DType.float32, [16, 10, 7], null, [], null);
    const output = stackedRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([16, 2]);

    // 3 trainable weights from each cell.
    expect(stackedRNN.trainableWeights.length).toEqual(6);
    expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
    // Kernel, recurrent kernel and bias of 1st cell.
    expect(stackedRNN.getWeights()[0].shape).toEqual([7, 3]);
    expect(stackedRNN.getWeights()[1].shape).toEqual([3, 3]);
    expect(stackedRNN.getWeights()[2].shape).toEqual([3]);
    // Kernel, recurrent kernel and bias of 2nd cell.
    expect(stackedRNN.getWeights()[3].shape).toEqual([3, 2]);
    expect(stackedRNN.getWeights()[4].shape).toEqual([2, 2]);
    expect(stackedRNN.getWeights()[5].shape).toEqual([2]);
  });

  it('With LSTMCell', () => {
    const stackedRNN = tfl.layers.rnn({
      cell: tfl.layers.stackedRNNCells({
        cells: [
          tfl.layers.lstmCell({units: 3, recurrentInitializer: 'glorotNormal'}),
          tfl.layers.lstmCell({units: 2, recurrentInitializer: 'glorotNormal'})
        ],
      })
    });
    const input =
        new tfl.SymbolicTensor(DType.float32, [16, 10, 7], null, [], null);
    const output = stackedRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([16, 2]);

    // 3 trainable weights from each cell.
    expect(stackedRNN.trainableWeights.length).toEqual(6);
    expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
    // Kernel, recurrent kernel and bias of 1st cell.
    expect(stackedRNN.getWeights()[0].shape).toEqual([7, 12]);
    expect(stackedRNN.getWeights()[1].shape).toEqual([3, 12]);
    expect(stackedRNN.getWeights()[2].shape).toEqual([12]);
    // expect(stackedRNN.getWeights()[2].shape).toEqual([3]);
    // Kernel, recurrent kernel and bias of 2nd cell.
    expect(stackedRNN.getWeights()[3].shape).toEqual([3, 8]);
    expect(stackedRNN.getWeights()[4].shape).toEqual([2, 8]);
    expect(stackedRNN.getWeights()[5].shape).toEqual([8]);
  });

  it('RNN with cell array creates StackedRNNCell', () => {
    const stackedRNN = tfl.layers.rnn({
      cell: [
        tfl.layers.gruCell({units: 3, recurrentInitializer: 'glorotNormal'}),
        tfl.layers.gruCell({units: 2, recurrentInitializer: 'glorotNormal'}),
      ],
    });
    const input =
        new tfl.SymbolicTensor(DType.float32, [16, 10, 7], null, [], null);
    const output = stackedRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([16, 2]);

    // 3 trainable weights from each cell.
    expect(stackedRNN.trainableWeights.length).toEqual(6);
    expect(stackedRNN.nonTrainableWeights.length).toEqual(0);
    // Kernel, recurrent kernel and bias of 1st cell.
    expect(stackedRNN.getWeights()[0].shape).toEqual([7, 9]);
    expect(stackedRNN.getWeights()[1].shape).toEqual([3, 9]);
    expect(stackedRNN.getWeights()[2].shape).toEqual([9]);
    // expect(stackedRNN.getWeights()[2].shape).toEqual([3]);
    // Kernel, recurrent kernel and bias of 2nd cell.
    expect(stackedRNN.getWeights()[3].shape).toEqual([3, 6]);
    expect(stackedRNN.getWeights()[4].shape).toEqual([2, 6]);
    expect(stackedRNN.getWeights()[5].shape).toEqual([6]);
  });
});

describeMathGPU('StackedRNNCells Tensor', () => {
  // The golden values for assertion below can be obtained with the following
  // Python Keras code:
  //
  // ```python
  // import keras
  // import numpy as np
  //
  // stacked_rnn = keras.layers.RNN(
  //     [
  //       keras.layers.SimpleRNNCell(
  //           3,
  //           kernel_initializer='ones',
  //           recurrent_initializer='ones',
  //           use_bias=False),
  //       keras.layers.GRUCell(
  //           2,
  //           kernel_initializer='ones',
  //           recurrent_initializer='ones',
  //           use_bias=False),
  //       keras.layers.LSTMCell(
  //           1,
  //           kernel_initializer='ones',
  //           recurrent_initializer='ones',
  //           use_bias=False),
  //     ])
  //
  // t_input = keras.layers.Input(batch_shape=(2, 3, 4))
  // t_output = stacked_rnn(t_input)
  // print(t_input.shape)
  // print(t_output.shape)
  //
  // model = keras.Model(t_input, t_output)
  //
  // input_val = np.array([
  //     [
  //         [0.1, -0.1, 0.2, -0.2], [-0.1, 0.1, -0.2, 0.2],
  //         [0.1, 0.1, -0.2, -0.2]
  //     ],
  //     [
  //         [0.05, -0.05, 0.1, -0.1], [-0.05, 0.05, -0.1, 0.1],
  //         [0.05, 0.05, -0.1, -0.1]
  //     ]
  // ])
  // print(model.predict(input_val))
  // ```
  it('Forward pass', () => {
    const stackedRNN = tfl.layers.rnn({
      cell: tfl.layers.stackedRNNCells({
        cells: [
          tfl.layers.simpleRNNCell({
            units: 3,
            recurrentInitializer: 'ones',
            kernelInitializer: 'ones',
            useBias: false
          }),
          tfl.layers.gruCell({
            units: 2,
            recurrentInitializer: 'ones',
            kernelInitializer: 'ones',
            useBias: false
          }),
          tfl.layers.lstmCell({
            units: 1,
            recurrentInitializer: 'ones',
            kernelInitializer: 'ones',
            useBias: false
          }),
        ],
      })
    });
    const input = tensor3d(
        [
          [
            [0.1, -0.1, 0.2, -0.2], [-0.1, 0.1, -0.2, 0.2],
            [0.1, 0.1, -0.2, -0.2]
          ],
          [
            [0.05, -0.05, 0.1, -0.1], [-0.05, 0.05, -0.1, 0.1],
            [0.05, 0.05, -0.1, -0.1]
          ]
        ],
        [2, 3, 4]);
    const output = stackedRNN.apply(input) as Tensor;
    expectTensorsClose(
        output, tensor2d([[-0.07715216], [-0.05906887]], [2, 1]));
  });
});
