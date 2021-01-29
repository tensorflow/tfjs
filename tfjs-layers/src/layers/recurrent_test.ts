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

import * as tfc from '@tensorflow/tfjs-core';
import {io, randomNormal, scalar, Tensor, tensor1d, tensor2d, tensor3d, tensor4d} from '@tensorflow/tfjs-core';

import {serializeActivation, Tanh} from '../activations';
import * as K from '../backend/tfjs_backend';
import {NonNeg, serializeConstraint, UnitNorm} from '../constraints';
import * as tfl from '../index';
import {GlorotUniform, HeUniform, Ones, serializeInitializer} from '../initializers';
import {ActivationIdentifier} from '../keras_format/activation_config';
import {ModelAndWeightsConfig, modelFromJSON} from '../models';
import {L1L2, serializeRegularizer} from '../regularizers';
import {Kwargs} from '../types';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, describeMathGPU, expectTensorsClose} from '../utils/test_utils';

import {GRUCellLayerArgs, GRULayerArgs, LSTMCellLayerArgs, LSTMLayerArgs, rnn, RNN, RNNCell, SimpleRNNCellLayerArgs, SimpleRNNLayerArgs} from './recurrent';

/**
 * A simplistic RNN step function for testing.
 * This step function simply
 * - calculates a reduced mean over all input elements, for each sample.
 * - adds that mean to the state tensor(s),
 * - take the negative of the 1st current state tensor and use it as the
 *   output.
 * @param inputs
 * @param states
 */
function rnnStepForTest(inputs: Tensor, states: Tensor[]): [Tensor, Tensor[]] {
  const mean = tfc.mean(inputs);
  const newStates = states.map(state => tfc.add(mean, state));
  const output = tfc.neg(newStates[0]);
  return [output, newStates];
}

describeMathCPUAndGPU('rnn', () => {
  it('Simple step function: 3D inputs, 1 state', () => {
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const initialStates = [tfc.zeros([2, 4])];
    const rnnOutputs =
        rnn(rnnStepForTest, inputs, initialStates, false /* goBackwards */,
            null /* mask */, null /* constants */, false /* unroll */,
            true /* needPerStepOutputs */);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75], [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(1);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
  });

  it('Simple step function: 3D inputs, 2 states', () => {
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    // The two state tensors have different shapes.
    const initialStates = [tfc.zeros([2, 4]), tfc.ones([2, 3])];
    const rnnOutputs =
        rnn(rnnStepForTest, inputs, initialStates, false /* goBackwards */,
            null /* mask */, null /* constants */, false /* unroll */,
            true /* needPerStepOutputs */);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75], [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(2);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
    expectTensorsClose(
        newStates[1],
        tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
  });

  it('Simple step function: 4D inputs, 2 states', () => {
    const inputs = tensor4d(
        [
          [[[1], [2]], [[3], [4]], [[5], [6]]],
          [[[10], [20]], [[30], [40]], [[50], [60]]]
        ],
        [2, 3, 2, 1]);
    // The two state tensors have different shapes.
    const initialStates = [tfc.zeros([2, 4]), tfc.ones([2, 3])];
    const rnnOutputs =
        rnn(rnnStepForTest, inputs, initialStates, false /* goBackwards */,
            null /* mask */, null /* constants */, false /* unroll */,
            true /* needPerStepOutputs */);
    const lastOutput = rnnOutputs[0];
    const outputs = rnnOutputs[1];
    const newStates = rnnOutputs[2];
    expectTensorsClose(
        lastOutput,
        tensor2d(
            [
              [-57.75, -57.75, -57.75, -57.75], [-57.75, -57.75, -57.75, -57.75]
            ],
            [2, 4]));
    expectTensorsClose(
        outputs,
        tensor3d(
            [
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ],
              [
                [-8.25, -8.25, -8.25, -8.25], [-27.5, -27.5, -27.5, -27.5],
                [-57.75, -57.75, -57.75, -57.75]
              ]
            ],
            [2, 3, 4]));
    expect(newStates.length).toEqual(2);
    expectTensorsClose(
        newStates[0],
        tensor2d(
            [[57.75, 57.75, 57.75, 57.75], [57.75, 57.75, 57.75, 57.75]],
            [2, 4]));
    expectTensorsClose(
        newStates[1],
        tensor2d([[58.75, 58.75, 58.75], [58.75, 58.75, 58.75]], [2, 3]));
  });

  it('Using inputs <3D leads to ValueError', () => {
    const inputs = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const initialStates = [tfc.zeros([4]), tfc.ones([3])];
    expect(() => rnn(rnnStepForTest, inputs, initialStates)).toThrowError();
  });
});

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
  /** @nocollapse */
  static className = 'RNNCellForTest';
  stateSize: number|number[];
  constructor(stateSizes: number|number[]) {
    super({});
    this.stateSize = stateSizes;
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    inputs = inputs as Tensor[];
    const dataInputs = inputs[0];
    const states = inputs.slice(1);
    const mean = tfc.mean(dataInputs);
    const newStates = states.map(state => tfc.add(mean, state));
    const output = tfc.neg(newStates[0]);
    return [output].concat(newStates);
  }
}

describeMathCPU('RNN-Layer', () => {
  // TODO(cais): Add tests for stacked RNN cell (i.e., multiple cells) once it
  //   implemented.
  // TODO(cais): Add tests for masks once implemented.
  // TODO(cais): Add tests for constants once implemented.

  it('constructor: only cell', () => {
    const cell = new RNNCellForTest(5);
    const rnn = tfl.layers.rnn({cell});
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
    });
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
           new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor;
       expect(output.shape).toEqual([16, 6]);
     });

  it('apply: Symbolic: 1 state, returnSequences=true, returnState=false',
     () => {
       const cell = new RNNCellForTest(6);
       const rnn = tfl.layers.rnn({cell, returnSequences: true});
       const input =
           new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
       const output = rnn.apply(input) as tfl.SymbolicTensor;
       expect(output.shape).toEqual([16, 10, 6]);
     });

  it('apply: Symbolic: 1 state, returnSequences=true, returnState=true', () => {
    const cell = new RNNCellForTest(6);
    const rnn =
        tfl.layers.rnn({cell, returnSequences: true, returnState: true});
    const input =
        new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
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
           new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
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
           new tfl.SymbolicTensor('float32', [16, 10, 8], null, [], null);
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
    const inputs = tfc.zeros([4, 3, 2]);
    const rnn = tfl.layers.rnn({cell});
    const initialStates = rnn.getInitialState(inputs);
    expect(initialStates.length).toEqual(1);
    expectTensorsClose(initialStates[0], tfc.zeros([4, 5]));
  });

  it('getInitialState: 2 states', () => {
    const cell = new RNNCellForTest([5, 6]);
    const inputs = tfc.zeros([4, 3, 2]);
    const rnn = tfl.layers.rnn({cell});
    const initialStates = rnn.getInitialState(inputs);
    expect(initialStates.length).toEqual(2);
    expectTensorsClose(initialStates[0], tfc.zeros([4, 5]));
    expectTensorsClose(initialStates[1], tfc.zeros([4, 6]));
  });

  it('call: 1 state: returnSequences=false, returnState=false', () => {
    const cell = new RNNCellForTest(4);
    const rnn = tfl.layers.rnn({cell});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs) as Tensor;
    expectTensorsClose(outputs, tfc.mul(scalar(-57.75), tfc.ones([2, 4])));
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
        rnn.apply(inputs, {'initialState': [tfc.ones([2, 4])]}) as Tensor;
    expectTensorsClose(outputs, tfc.mul(scalar(-58.75), tfc.ones([2, 4])));
  });

  it('call: with 2 initialStates', () => {
    const cell = new RNNCellForTest([4, 5]);
    const rnn = tfl.layers.rnn({cell, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    const outputs = rnn.apply(inputs, {
      'initialState': [tfc.ones([2, 4]), tfc.mul(scalar(2), tfc.ones([2, 5]))]
    }) as Tensor[];
    expect(outputs.length).toEqual(3);
    expectTensorsClose(outputs[0], tfc.mul(scalar(-58.75), tfc.ones([2, 4])));
    expectTensorsClose(outputs[1], tfc.mul(scalar(58.75), tfc.ones([2, 4])));
    expectTensorsClose(outputs[2], tfc.mul(scalar(59.75), tfc.ones([2, 5])));
  });

  it('call with incorrect number of initialStates leads to ValueError', () => {
    const cell = new RNNCellForTest([4, 5]);
    const rnn = tfl.layers.rnn({cell, returnState: true});
    const inputs = tensor3d(
        [[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]], [2, 3, 2]);
    expect(() => rnn.apply(inputs, {
      'initialState': [tfc.ones([2, 4])]
    })).toThrowError(/An initialState was passed that is not compatible with/);
  });
});

describeMathCPU('SimpleRNN Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5, returnState: true});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const simpleRNN = tfl.layers.simpleRNN({units: 5, returnSequences: true});
    const output = simpleRNN.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
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

  it('Serialization round trip', () => {
    const layer = tfl.layers.simpleRNN({units: 4});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.simpleRNN(tsConfig);
    expect(layerPrime.getConfig().units).toEqual(4);
  });

  it('Invalid units leads to Error', () => {
    expect(() => tfl.layers.simpleRNN({
      units: 12.5
    })).toThrowError(/units.*positive integer.*12\.5\.$/);
    expect(() => tfl.layers.simpleRNN({
      units: 0
    })).toThrowError(/units.*positive integer.*0\.$/);
    expect(() => tfl.layers.simpleRNN({
      units: -25
    })).toThrowError(/units.*positive integer.*-25\.$/);
  });
});

describeMathCPUAndGPU('SimpleRNN Tensor', () => {
  const units = 5;
  const batchSize = 4;
  const inputSize = 2;

  // TODO(cais): Add test for the default recurrent initializer ('Orthogonal')
  //   when it becomes available.

  const dropouts = [0.0, 0.1];
  const trainings = [true, false];
  for (const training of trainings) {
    for (const dropout of dropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, dropout=${dropout}`;
      it(testTitle, () => {
        const timeSteps = 3;
        const simpleRNN = tfl.layers.simpleRNN({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          dropout,
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(simpleRNN.apply(input, kwargs) as Tensor);
          if (dropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(1 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

  const recurrentDropouts = [0.0, 0.1];
  for (const training of trainings) {
    for (const recurrentDropout of recurrentDropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, recurrentDropout=${recurrentDropout}`;
      it(testTitle, () => {
        const timeSteps = 3;
        const simpleRNN = tfl.layers.simpleRNN({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          recurrentDropout,
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(simpleRNN.apply(input, kwargs) as Tensor);
          if (recurrentDropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(1 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

  const activations: ActivationIdentifier[] = ['linear', 'tanh'];
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
      const input = tfc.ones([batchSize, timeSteps, inputSize]);
      const output = simpleRNN.apply(input) as Tensor;
      let expectedElementValue = inputSize + 1;
      if (activation === 'tanh') {
        expectedElementValue = Math.tanh(expectedElementValue);
      }
      expectTensorsClose(
          output,
          tfc.mul(scalar(expectedElementValue), tfc.ones([batchSize, units])));
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
      const input = tfc.ones([batchSize, timeSteps, inputSize]);
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
      const timeMajorOutput = tfc.transpose(output, [1, 0, 2]);
      const outputT0 = K.sliceAlongFirstAxis(timeMajorOutput, 0, 1);
      const outputT1 = K.sliceAlongFirstAxis(timeMajorOutput, 1, 1);
      expectTensorsClose(
          outputT0,
          tfc.mul(scalar(inputSize + 1), tfc.ones([1, batchSize, units])));
      expectTensorsClose(
          outputT1,
          tfc.mul(
              scalar((inputSize + 1) * (units + 1)),
              tfc.ones([1, batchSize, units])));
      if (returnState) {
        expectTensorsClose(finalState, outputT1.reshape([batchSize, units]));
      }
    });

    it('stateful missing batchInputShape leads to error', () => {
      const sequenceLength = 3;
      const simpleRNN = tfl.layers.simpleRNN({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'zeros',
        activation: 'linear',
        stateful: true,
        inputShape: [sequenceLength, inputSize]
      }) as RNN;
      const model = tfl.sequential();
      expect(() => model.add(simpleRNN))
          .toThrowError(/needs to know its batch size/);
    });

    // The reference values below can be obtained with PyKeras code:
    // ```python
    // import keras
    // import numpy as np
    //
    // units = 5
    // batch_size = 4
    // input_size = 2
    // sequence_length = 3
    //
    // rnn = keras.layers.SimpleRNN(units,
    //                             kernel_initializer='ones',
    //                             recurrent_initializer='ones',
    //                             bias_initializer='zeros',
    //                             activation='linear',
    //                             stateful=True,
    //                             batch_input_shape=[batch_size,
    //                                                 sequence_length,
    //                                                 input_size])
    //
    // x = np.ones([batch_size, sequence_length, input_size])
    //
    // model = keras.Sequential()
    // model.add(rnn)
    // print(model.predict(x))
    // print(model.predict(x))
    // model.reset_states()
    // print(model.predict(x))
    // print(model.predict(x))
    // ```
    it('stateful forward: only RNN layer', async () => {
      const sequenceLength = 3;
      const rnn = tfl.layers.simpleRNN({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'zeros',
        activation: 'linear',
        stateful: true,
        batchInputShape: [batchSize, sequenceLength, inputSize]
      });
      const model = tfl.sequential();
      model.add(rnn);
      const x = tfc.ones([batchSize, sequenceLength, inputSize]);
      const scalar1 = tfc.scalar(62);
      const scalar2 = tfc.scalar(7812);

      let y1 = model.predict(x) as Tensor;
      expect(y1.kept).toBe(false);
      let y1Expected =
          tfc.tidy(() => tfc.ones([batchSize, units]).mul(scalar1));
      expectTensorsClose(y1, y1Expected);

      let y2 = model.predict(x) as Tensor;
      expect(y2.kept).toBe(false);
      // Future predicts should not dispose previous outputs.
      expect(y1.isDisposed).toBe(false);
      let y2Expected =
          tfc.tidy(() => tfc.ones([batchSize, units]).mul(scalar2));
      expectTensorsClose(y2, y2Expected);
      tfc.dispose([y1, y2, y1Expected, y2Expected]);

      model.resetStates();
      const numTensors0 = tfc.memory().numTensors;

      y1 = model.predict(x) as Tensor;
      expect(y1.kept).toBe(false);
      y1Expected = tfc.tidy(() => tfc.ones([batchSize, units]).mul(scalar1));
      expectTensorsClose(y1, y1Expected);

      y2 = model.predict(x) as Tensor;
      expect(y2.kept).toBe(false);
      // Future predicts should not dispose previous outputs.
      expect(y1.isDisposed).toBe(false);
      y2Expected = tfc.tidy(() => tfc.ones([batchSize, units]).mul(scalar2));
      expectTensorsClose(y2, y2Expected);
      tfc.dispose([y1, y2, y1Expected, y2Expected]);

      // Assert no memory leak, even without resetStates() being called.
      expect(tfc.memory().numTensors).toEqual(numTensors0);
    });
  }

  // Reference Python code:
  // ```py
  // import numpy as np
  // from tensorflow import keras
  //
  // model = keras.Sequential()
  // model.add(keras.layers.SimpleRNN(
  //     units=5,
  //     kernel_initializer='ones',
  //     recurrent_initializer='ones',
  //     bias_initializer='zeros',
  //     activation='linear',
  //     stateful=True,
  //     batch_input_shape=[4, 3, 2]))
  // model.add(keras.layers.Dense(units=1, kernel_initializer='ones'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  // model.summary()
  //
  // xs = np.ones([4, 3, 2])
  // y1 = model.predict(xs)
  // print(y1)
  // y2 = model.predict(xs)
  // print(y2)
  //
  // history = model.fit(xs, ys, batch_size=4, epochs=3)
  // print(history.history)
  // ```
  it('stateful forward: RNN and dense layers', async () => {
    const sequenceLength = 3;
    const model = tfl.sequential();
    model.add(tfl.layers.simpleRNN({
      units,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'zeros',
      activation: 'linear',
      stateful: true,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    }));
    model.add(tfl.layers.dense({units: 1, kernelInitializer: 'ones'}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    const xs = tfc.ones([batchSize, sequenceLength, inputSize]);
    const ys = tfc.ones([batchSize, 1]);
    let y1: Tensor;
    let y2: Tensor;
    tfc.tidy(() => {
      y1 = model.predict(xs) as Tensor;
      expectTensorsClose(y1, tfc.tensor2d([310, 310, 310, 310], [4, 1]));
      y2 = model.predict(xs) as Tensor;
      expectTensorsClose(
          y2, tfc.tensor2d([39060, 39060, 39060, 39060], [4, 1]));
    });
    model.resetStates();
    const numTensors0 = tfc.memory().numTensors;

    tfc.tidy(() => {
      y1 = model.predict(xs) as Tensor;
      expect(y1.shape).toEqual([batchSize, 1]);
      expectTensorsClose(y1, tfc.tensor2d([310, 310, 310, 310], [4, 1]));
      y2 = model.predict(xs) as Tensor;
      expectTensorsClose(
          y2, tfc.tensor2d([39060, 39060, 39060, 39060], [4, 1]));
    });
    // Assert no memory leak, even without resetStates() being called.
    expect(tfc.memory().numTensors).toEqual(numTensors0);

    const history = await model.fit(xs, ys, {epochs: 1, batchSize: 4});
    expect(history.history.loss[0]).toBeCloseTo(23841822736384);
  });

  it('computeMask: returnSequence = false, returnState = false', () => {
    const sequenceLength = 3;
    const rnn = tfl.layers.simpleRNN({
      units,
      returnSequences: false,
      returnState: false,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    });
    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    const m = tfc.ones([sequenceLength, 1]);
    const mask = rnn.computeMask(x, m);
    expect(mask).toBeNull();
  });

  it('computeMask: returnSequence = true, returnState = false', () => {
    const sequenceLength = 3;
    const rnn = tfl.layers.simpleRNN({
      units,
      returnSequences: true,
      returnState: false,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    });
    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    const m = tfc.ones([sequenceLength, 1]);
    const mask = rnn.computeMask(x, m) as Tensor;
    expectTensorsClose(mask, m);
  });

  it('computeMask: returnSequence = true, returnState = true', () => {
    const sequenceLength = 3;
    const rnn = tfl.layers.simpleRNN({
      units,
      returnSequences: true,
      returnState: true,
      stateful: true,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    });
    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    rnn.apply(x);
    rnn.resetStates();  // Let the RNN layer object construct its state first.
    const m = tfc.ones([sequenceLength, 1]);
    const masks = rnn.computeMask(x, m) as Tensor[];
    expect(masks.length).toEqual(2);
    expectTensorsClose(masks[0], m);
    expect(masks[1]).toBeNull();
  });

  // The reference values can be obtained with the following PyKeras code:
  // ```python
  // import keras
  // import numpy as np
  //
  // batch_size = 4
  // sequence_length = 3
  // input_size = 2
  //
  // model = keras.Sequential()
  // model.add(keras.layers.SimpleRNN(
  //     units=5,
  //     batch_input_shape=[batch_size, sequence_length, input_size],
  //     kernel_initializer='ones',
  //     recurrent_initializer='ones',
  //     bias_initializer='zeros',
  //     stateful=True))
  // model.add(keras.layers.Dense(1,
  //                              kernel_initializer='zeros',
  //                              use_bias=False))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs_1 = np.ones([batch_size, sequence_length, input_size])
  // xs_2 = np.zeros([batch_size, sequence_length, input_size])
  // xs = np.concatenate([xs_1, xs_2], 0)
  // ys = np.array([[-1], [-2], [0], [1], [1], [2], [0], [-1]])
  //
  // history = model.fit(xs, ys, batch_size=batch_size, shuffle=False)
  // print(history.history)
  // ```
  it('stateful BPTT', async () => {
    const sequenceLength = 3;
    const model = tfl.sequential();
    model.add(tfl.layers.simpleRNN({
      units,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'zeros',
      stateful: true,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tfc.ones([batchSize, sequenceLength, inputSize]);
    const xs2 = tfc.ones([batchSize, sequenceLength, inputSize]);
    const xs = tfc.concat([xs1, xs2], 0);
    const ys = tfc.tensor2d([[-1], [-2], [0], [1], [1], [2], [0], [-1]]);

    const history = await model.fit(xs, ys, {batchSize, shuffle: false});
    // See code snippet above for the code used to obtain this loss value.
    expect(history.history.loss[0]).toBeCloseTo(1.5262475);
  });

  it('BPTT', async () => {
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
    // optimizer = keras.optimizers.SGD(5)
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
    const model = tfl.sequential();
    model.add(tfl.layers.simpleRNN({
      units: 1,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      useBias: false,
      inputShape: [sequenceLength, inputSize],
    }));
    model.add(tfl.layers.dense({
      units: 1,
      kernelInitializer: 'ones',
      useBias: false,
    }));

    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    const y = tfc.zeros([batchSize, 1]);
    const sgd = tfc.train.sgd(5);
    model.compile({loss: 'meanSquaredError', optimizer: sgd});
    await model.fit(x, y, {epochs: 2});

    expectTensorsClose(
        model.layers[0].getWeights()[0],
        tfc.mul(scalar(0.8484658), tfc.ones([4, 1])));
    expectTensorsClose(
        model.layers[0].getWeights()[1],
        tfc.mul(scalar(0.8484799), tfc.ones([1, 1])));
    expectTensorsClose(
        model.layers[1].getWeights()[0],
        tfc.mul(scalar(80.967026), tfc.ones([1, 1])));
  });
});

describeMathCPU('SimpleRNN Serialization', () => {
  const cellConfig: SimpleRNNCellLayerArgs = {
    units: 8,
    activation: 'tanh',
    useBias: true,
    kernelInitializer: 'glorotUniform',
    recurrentInitializer: 'heUniform',
    biasInitializer: 'ones',
    kernelRegularizer: 'l1l2',
    recurrentRegularizer: 'l1l2',
    biasRegularizer: 'l1l2',
    kernelConstraint: 'unitNorm',
    recurrentConstraint: 'unitNorm',
    biasConstraint: 'nonNeg',
    dropout: 0.1,
    recurrentDropout: 0.2,
    name: 'cell_1',
    batchSize: 12,
    batchInputShape: [12, 8, 8],
    inputShape: [8, 8],
    dtype: 'int32',
    inputDType: 'int32',
    trainable: true,
  };

  const expectedCellConfigPrime = {
    name: 'cell_1',
    trainable: true,
    batchInputShape: [12, 8, 8],
    dtype: 'int32',
    units: 8,
    activation: serializeActivation(new Tanh()),
    useBias: true,
    kernelInitializer: serializeInitializer(new GlorotUniform()),
    recurrentInitializer: serializeInitializer(new HeUniform()),
    biasInitializer: serializeInitializer(new Ones()),
    kernelRegularizer: serializeRegularizer(new L1L2()),
    recurrentRegularizer: serializeRegularizer(new L1L2()),
    biasRegularizer: serializeRegularizer(new L1L2()),
    activityRegularizer: serializeRegularizer(null),
    kernelConstraint: serializeConstraint(new UnitNorm({})),
    recurrentConstraint: serializeConstraint(new UnitNorm({})),
    biasConstraint: serializeConstraint(new NonNeg()),
  };

  describe('SimpleRNNCell.getConfig', () => {
    it('should return the expected values', () => {
      const cell = tfl.layers.simpleRNNCell(cellConfig);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual(expectedCellConfigPrime);
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });

  describe('SimpleRNN.getConfig', () => {
    it('should return the expected values', () => {
      const config: SimpleRNNLayerArgs = {
        ...cellConfig,
        name: 'layer_1',
        ...{
          returnSequences: true,
          returnState: true,
          stateful: true,
          unroll: true,
          goBackwards: true,
          inputDim: 8,
          inputLength: 8,
        } as Omit<SimpleRNNLayerArgs, keyof SimpleRNNCellLayerArgs>
      };

      const cell = tfl.layers.simpleRNN(config);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual({
        ...expectedCellConfigPrime,
        name: 'layer_1',
        returnSequences: true,
        returnState: true,
        stateful: true,
        unroll: true,
        goBackwards: true,
      });
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });
});

describeMathCPU('GRU Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5});
    const output = gru.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5, returnState: true});
    const output = gru.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(2);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const gru = tfl.layers.gru({units: 5, returnSequences: true});
    const output = gru.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
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
           new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
       const gru = tfl.layers.gru({units: 5, returnState: true});
       gru.apply(input);
       expect(gru.trainable).toEqual(true);
       // Trainable weights: kernel, recurrent kernel and bias.
       expect(gru.trainableWeights.length).toEqual(3);
       expect(gru.nonTrainableWeights.length).toEqual(0);
       expect(gru.weights.length).toEqual(3);
     });

  for (const implementation of [1, 2]) {
    it('Serialization round trip', () => {
      const layer = tfl.layers.gru({units: 4, implementation});
      const pythonicConfig = convertTsToPythonic(layer.getConfig());
      // tslint:disable-next-line:no-any
      const tsConfig = convertPythonicToTs(pythonicConfig) as any;
      const layerPrime = tfl.layers.gru(tsConfig);
      expect(layerPrime.getConfig().units).toEqual(4);
      expect(layerPrime.getConfig().implementation).toEqual(implementation);
    });
  }

  it('Invalid units leads to Error', () => {
    expect(() => tfl.layers.gru({
      units: 12.5
    })).toThrowError(/units.*positive integer.*12\.5\.$/);
    expect(() => tfl.layers.gru({
      units: 0
    })).toThrowError(/units.*positive integer.*0\.$/);
    expect(() => tfl.layers.gru({
      units: -25
    })).toThrowError(/units.*positive integer.*-25\.$/);
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

  const dropouts = [0.0, 0.1];
  const trainings = [true, false];
  for (const training of trainings) {
    for (const dropout of dropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, dropout=${dropout}`;
      it(testTitle, () => {
        const gru = tfl.layers.gru({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          dropout,
          implementation: 1
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(gru.apply(input, kwargs) as Tensor);
          if (dropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(3 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

  const recurrentDropouts = [0.0, 0.1];
  for (const training of trainings) {
    for (const recurrentDropout of recurrentDropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, recurrentDropout=${recurrentDropout}`;
      it(testTitle, () => {
        const gru = tfl.layers.gru({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          recurrentDropout,
          implementation: 1
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(gru.apply(input, kwargs) as Tensor);
          if (recurrentDropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(3 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

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
          const input = tfc.zeros([batchSize, timeSteps, inputSize]);
          let output = gru.apply(input);

          const goldenOutputElementValueFinal =
              goldenOutputElementValues[goldenOutputElementValues.length - 1];

          let expectedOutput: Tensor;
          if (returnSequences) {
            const outputs = goldenOutputElementValues.map(
                value =>
                    tfc.mul(scalar(value), tfc.ones([1, batchSize, units])));
            expectedOutput = tfc.transpose(
                K.concatAlongFirstAxis(
                    K.concatAlongFirstAxis(outputs[0], outputs[1]), outputs[2]),
                [1, 0, 2]);
          } else {
            expectedOutput = tfc.mul(
                scalar(goldenOutputElementValueFinal),
                tfc.ones([batchSize, units]));
          }
          if (returnState) {
            output = output as Tensor[];
            expect(output.length).toEqual(2);
            expectTensorsClose(output[0], expectedOutput);
            expectTensorsClose(
                output[1],
                tfc.mul(
                    scalar(goldenOutputElementValueFinal),
                    tfc.ones([batchSize, units])));
          } else {
            output = output as Tensor;
            expectTensorsClose(output, expectedOutput);
          }
        });
      }
    }
  }

  // The reference values below can be obtained with PyKeras code:
  // ```python
  // import keras
  // import numpy as np
  //
  // units = 5
  // batch_size = 4
  // input_size = 2
  // sequence_length = 3
  //
  // rnn = keras.layers.GRU(units,
  //                        kernel_initializer='ones',
  //                        recurrent_initializer='zeros',
  //                        bias_initializer='zeros',
  //                        activation='linear',
  //                        stateful=True,
  //                        batch_input_shape=[batch_size,
  //                                           sequence_length,
  //                                           input_size])
  //
  // x = np.ones([batch_size, sequence_length, input_size])
  //
  // model = keras.Sequential()
  // model.add(rnn)
  // print(model.predict(x))
  // print(model.predict(x))
  // model.reset_states()
  // print(model.predict(x))
  // print(model.predict(x))
  // ```
  it('stateful forward', async () => {
    const sequenceLength = 3;
    const rnn = tfl.layers.gru({
      units,
      kernelInitializer: 'ones',
      recurrentInitializer: 'zeros',
      biasInitializer: 'zeros',
      activation: 'linear',
      stateful: true,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    });
    const model = tfl.sequential();
    model.add(rnn);
    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    const y1 = model.predict(x) as Tensor;
    expect(y1.kept).toBe(false);
    const y1Expected =
        tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.542)));
    expectTensorsClose(y1, y1Expected);
    const y2 = model.predict(x) as Tensor;
    expect(y2.kept).toBe(false);
    // Future predicts should not dispose previous outputs.
    expect(y1.isDisposed).toBe(false);
    const y2Expected =
        tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.9371182)));
    expectTensorsClose(y2, y2Expected);

    tfc.dispose([y1, y2, y1Expected, y2Expected]);
    model.resetStates();
    const numTensors0 = tfc.memory().numTensors;

    const y3 = model.predict(x) as Tensor;
    expect(y3.kept).toBe(false);
    const y3Expected =
        tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.542)));
    expectTensorsClose(y3, y3Expected);
    const y4 = model.predict(x) as Tensor;
    expect(y4.kept).toBe(false);
    // Future predicts should not dispose previous outputs.
    expect(y3.isDisposed).toBe(false);
    const y4Expected =
        tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.9371182)));
    expectTensorsClose(y4, y4Expected);
    tfc.dispose([y3, y3Expected, y4, y4Expected]);
    // Assert no memory leak, even without resetStates() being called.
    expect(tfc.memory().numTensors).toEqual(numTensors0);
  });

  // The reference values can be obtained with the following PyKeras code:
  // ```python
  // import keras
  // import numpy as np
  //
  // batch_size = 4
  // sequence_length = 3
  // input_size = 2
  //
  // model = keras.Sequential()
  // model.add(keras.layers.GRU(
  //     units=5,
  //     batch_input_shape=[batch_size, sequence_length, input_size],
  //     kernel_initializer='ones',
  //     recurrent_initializer='ones',
  //     bias_initializer='zeros',
  //     stateful=True))
  // model.add(keras.layers.Dense(1,
  //                              kernel_initializer='zeros',
  //                              use_bias=False))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs_1 = np.ones([batch_size, sequence_length, input_size])
  // xs_2 = np.zeros([batch_size, sequence_length, input_size])
  // xs = np.concatenate([xs_1, xs_2], 0)
  // ys = np.array([[-1], [-2], [0], [1], [1], [2], [0], [-1]])
  //
  // history = model.fit(xs, ys, batch_size=batch_size, shuffle=False)
  // print(history.history)
  // ```
  it('stateful BPTT', async () => {
    const sequenceLength = 3;
    const model = tfl.sequential();
    model.add(tfl.layers.gru({
      units,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'zeros',
      stateful: true,
      batchInputShape: [batchSize, sequenceLength, inputSize]
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'zeros', useBias: false}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs1 = tfc.ones([batchSize, sequenceLength, inputSize]);
    const xs2 = tfc.ones([batchSize, sequenceLength, inputSize]);
    const xs = tfc.concat([xs1, xs2], 0);
    const ys = tfc.tensor2d([[-1], [-2], [0], [1], [1], [2], [0], [-1]]);

    const history = await model.fit(xs, ys, {batchSize, shuffle: false});
    // See code snippet above for the code used to obtain this loss value.
    expect(history.history.loss[0]).toBeCloseTo(1.501);
  });

  it('BPTT', async () => {
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
    const model = tfl.sequential();
    model.add(tfl.layers.gru({
      units: 1,
      kernelInitializer: 'zeros',
      recurrentInitializer: 'zeros',
      useBias: false,
      inputShape: [sequenceLength, inputSize]
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', useBias: false}));

    const sgd = tfc.train.sgd(1);
    model.compile({loss: 'meanSquaredError', optimizer: sgd});
    const x = tfc.ones([batchSize, sequenceLength, inputSize]);
    const y = tfc.ones([batchSize, 1]);
    await model.fit(x, y, {epochs: 2});
    expectTensorsClose(
        model.layers[0].getWeights()[0],
        K.tile(tensor2d([[-0.03750037, 0, 1.7500007]], [1, 3]), [4, 1]));
    expectTensorsClose(
        model.layers[0].getWeights()[1],
        tensor2d([[-1.562513e-02, 0, 2.086183e-07]], [1, 3]));
    expectTensorsClose(
        model.layers[1].getWeights()[0], tensor2d([[1.2187521]], [1, 1]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // in1 = keras.Input(shape=[5])
  // out1 = keras.layers.Dense(2,
  //                           kernel_initializer='ones',
  //                           bias_initializer='zeros')(in1)
  // in2 = keras.Input(shape=[3, 4])
  // out2 = keras.layers.GRU(2,
  //                         recurrent_initializer='ones',
  //                         kernel_initializer='ones',
  //                         bias_initializer='zeros')(in2, initial_state=out1)
  // model = keras.Model(inputs=[in1, in2], outputs=[out1, out2])
  //
  // xs1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
  // xs2 = np.array([[[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4],
  //                 [0.3, 0.4, 0.5, 0.6]]])
  // print(model.predict([xs1, xs2]))
  // ```
  it('SymbolicTensor as initialState thru kwargs; Save & Load', async () => {
    const in1 = tfl.input({shape: [5]});
    const out1 =
        tfl.layers
            .dense(
                {units: 2, kernelInitializer: 'ones', biasInitializer: 'zeros'})
            .apply(in1) as tfl.SymbolicTensor;
    const in2 = tfl.input({shape: [3, 4]});
    const out2 = tfl.layers
                     .gru({
                       units: 2,
                       recurrentInitializer: 'ones',
                       kernelInitializer: 'ones',
                       biasInitializer: 'zeros'
                     })
                     .apply(in2, {initialState: out1}) as tfl.SymbolicTensor;

    const model = tfl.model({inputs: [in1, in2], outputs: [out1, out2]});

    const xs1 = tensor2d([[0.1, 0.2, 0.3, 0.4, 0.5]]);
    const xs2 = tensor3d([
      [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4], [0.3, 0.4, 0.5, 0.6]]
    ]);
    const ys = model.predict([xs1, xs2]) as Tensor[];
    expect(ys.length).toEqual(2);
    expectTensorsClose(ys[0], tensor2d([[1.5, 1.5]]));
    expectTensorsClose(ys[1], tensor2d([[1.4435408, 1.4435408]]));

    // NOTE: Here on down, i.e., the part that tests serialization and
    // deserialization of the model, has no counterpart in the Python
    // code snippet above.
    const modelJSON = model.toJSON(null, false);
    const modelPrime =
        await tfl.models.modelFromJSON({modelTopology: modelJSON});
    const ysPrime = modelPrime.predict([xs1, xs2]) as Tensor[];
    expect(ysPrime.length).toEqual(2);
    expectTensorsClose(ysPrime[0], ys[0]);
    expectTensorsClose(ysPrime[1], ys[1]);
  });
});

describeMathCPU('GRU-deserialization', () => {
  it('Default recurrentActivation round trip', () => {
    const x = randomNormal([1, 2, 3]);
    const layer = tfl.layers.gru({units: 4});
    const y = layer.apply(x) as Tensor;
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.gru(tsConfig);
    const yPrime = layer.apply(x) as Tensor;
    expectTensorsClose(yPrime, y);
    expect(layerPrime.getConfig()['recurrentActivation'])
        .toEqual(layer.getConfig()['recurrentActivation']);
  });

  it('Non-default recurrentActivation round trip', () => {
    const x = randomNormal([1, 2, 3]);
    const layer =
        tfl.layers.gru({units: 4, recurrentActivation: 'tanh'});
    const y = layer.apply(x) as Tensor;
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.gru(tsConfig);
    const yPrime = layer.apply(x) as Tensor;
    expectTensorsClose(yPrime, y);
    expect(layerPrime.getConfig()['recurrentActivation'])
        .toEqual(layer.getConfig()['recurrentActivation']);
  });
});

describeMathCPU('GRU Serialization', () => {
  const cellConfig: GRUCellLayerArgs = {
    units: 8,
    activation: 'tanh',
    recurrentActivation: 'tanh',
    useBias: true,
    kernelInitializer: 'glorotUniform',
    recurrentInitializer: 'heUniform',
    biasInitializer: 'ones',
    kernelRegularizer: 'l1l2',
    recurrentRegularizer: 'l1l2',
    biasRegularizer: 'l1l2',
    kernelConstraint: 'unitNorm',
    recurrentConstraint: 'unitNorm',
    biasConstraint: 'nonNeg',
    dropout: 0.1,
    recurrentDropout: 0.2,
    name: 'cell_1',
    batchSize: 12,
    batchInputShape: [12, 8, 8],
    inputShape: [8, 8],
    dtype: 'int32',
    inputDType: 'int32',
    trainable: true,
    implementation: 1,
  };

  const expectedCellConfigPrime = {
    name: 'cell_1',
    trainable: true,
    batchInputShape: [12, 8, 8],
    dtype: 'int32',
    units: 8,
    activation: serializeActivation(new Tanh()),
    recurrentActivation: serializeActivation(new Tanh()),
    useBias: true,
    kernelInitializer: serializeInitializer(new GlorotUniform()),
    recurrentInitializer: serializeInitializer(new HeUniform()),
    biasInitializer: serializeInitializer(new Ones()),
    kernelRegularizer: serializeRegularizer(new L1L2()),
    recurrentRegularizer: serializeRegularizer(new L1L2()),
    biasRegularizer: serializeRegularizer(new L1L2()),
    activityRegularizer: serializeRegularizer(null),
    kernelConstraint: serializeConstraint(new UnitNorm({})),
    recurrentConstraint: serializeConstraint(new UnitNorm({})),
    biasConstraint: serializeConstraint(new NonNeg()),
    implementation: 1,
    resetAfter: false,
  };

  describe('GRUCell.getConfig', () => {
    it('should return the expected values', () => {
      const cell = tfl.layers.gruCell(cellConfig);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual(expectedCellConfigPrime);
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });

  describe('GRU.getConfig', () => {
    it('should return the expected values', () => {
      const config: GRULayerArgs = {
        ...cellConfig,
        name: 'layer_1',
        ...{
          returnSequences: true,
          returnState: true,
          stateful: true,
          unroll: true,
          goBackwards: true,
          inputDim: 8,
          inputLength: 8,
        } as Omit<GRULayerArgs, keyof GRUCellLayerArgs>
      };

      const cell = tfl.layers.gru(config);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual({
        ...expectedCellConfigPrime,
        name: 'layer_1',
        returnSequences: true,
        returnState: true,
        stateful: true,
        unroll: true,
        goBackwards: true,
      });
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });
});

describeMathCPU('LSTM Symbolic', () => {
  it('returnSequences=false, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5});
    const output = lstm.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 5]);
  });

  it('returnSequences=false, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5, returnState: true});
    const output = lstm.apply(input) as tfl.SymbolicTensor[];
    expect(output.length).toEqual(3);
    expect(output[0].shape).toEqual([9, 5]);
    expect(output[1].shape).toEqual([9, 5]);
    expect(output[2].shape).toEqual([9, 5]);
  });

  it('returnSequences=true, returnState=false', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
    const lstm = tfl.layers.lstm({units: 5, returnSequences: true});
    const output = lstm.apply(input) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([9, 10, 5]);
  });

  it('returnSequences=true, returnState=true', () => {
    const input = new tfl.SymbolicTensor('float32', [9, 10, 8], null, [], null);
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
           new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
       const lstm = tfl.layers.lstm({units: 5, returnState: true});
       lstm.apply(input);
       expect(lstm.trainable).toEqual(true);
       // Trainable weights: kernel, recurrent kernel and bias.
       expect(lstm.trainableWeights.length).toEqual(3);
       expect(lstm.nonTrainableWeights.length).toEqual(0);
       expect(lstm.weights.length).toEqual(3);
     });

  for (const implementation of [1, 2]) {
    it('Serialization round trip', () => {
      const layer = tfl.layers.lstm({units: 4, implementation});
      const pythonicConfig = convertTsToPythonic(layer.getConfig());
      // tslint:disable-next-line:no-any
      const tsConfig = convertPythonicToTs(pythonicConfig) as any;
      const layerPrime = tfl.layers.lstm(tsConfig);
      expect(layerPrime.getConfig().units).toEqual(4);
      expect(layerPrime.getConfig().implementation).toEqual(implementation);
    });
  }

  it('LSTM Cells save and load', async () => {
    const inputShape = [2, 3];
    const model = tfl.sequential();
    model.add(tfl.layers.dense({units: 1, inputShape}));
    const cells = [
      tfl.layers.lstmCell({units: 3}),
      tfl.layers.lstmCell({units: 4}),
    ];
    const rnn = tfl.layers.rnn({cell: cells, returnSequences: true});
    model.add(rnn);

    const numExamples = 5;
    const xs = randomNormal([numExamples].concat(inputShape));
    const ys = model.predict(xs) as Tensor;

    let savedArtifacts: io.ModelArtifacts;
    await model.save(tfc.io.withSaveHandler(async (artifacts) => {
      savedArtifacts = artifacts;
      return null;
    }));

    const loadedModel =
        await tfl.loadLayersModel(tfc.io.fromMemory(savedArtifacts));

    expect(model.inputs[0].shape).toEqual(loadedModel.inputs[0].shape);
    expect(model.outputs[0].shape).toEqual(loadedModel.outputs[0].shape);
    expectTensorsClose(loadedModel.predict(xs) as Tensor, ys);
  });

  it('Invalid units leads to Error', () => {
    expect(() => tfl.layers.lstm({
      units: 12.5
    })).toThrowError(/units.*positive integer.*12\.5\.$/);
    expect(() => tfl.layers.lstm({
      units: 0
    })).toThrowError(/units.*positive integer.*0\.$/);
    expect(() => tfl.layers.lstm({
      units: -25
    })).toThrowError(/units.*positive integer.*-25\.$/);
  });
});

describeMathCPUAndGPU('LSTM Tensor', () => {
  // Note:
  // The golden expected values used for assertions in these unit tests can be
  // obtained through running the Python code similar to the following
  // example. TensorFlow 1.5 was used to obtain the values.
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

  const dropouts = [0.0, 0.1];
  const trainings = [true, false];
  for (const training of trainings) {
    for (const dropout of dropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, dropout=${dropout}`;
      it(testTitle, () => {
        const lstm = tfl.layers.lstm({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          dropout,
          implementation: 1
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(lstm.apply(input, kwargs) as Tensor);
          if (dropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(4 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

  const recurrentDropouts = [0.0, 0.1];
  for (const training of trainings) {
    for (const recurrentDropout of recurrentDropouts) {
      const testTitle =
          `returnSequences=false, returnState=false, useBias=true,` +
          ` ${training}, recurrentDropout=${recurrentDropout}`;
      it(testTitle, () => {
        const lstm = tfl.layers.lstm({
          units,
          kernelInitializer: 'ones',
          recurrentInitializer: 'ones',
          biasInitializer: 'ones',
          recurrentDropout,
          implementation: 1
        });
        const kwargs: Kwargs = {};
        if (training) {
          kwargs['training'] = true;
        }
        const input = tfc.ones([batchSize, timeSteps, inputSize]);
        spyOn(tfc, 'dropout').and.callThrough();
        let numTensors = 0;
        for (let i = 0; i < 2; i++) {
          tfc.dispose(lstm.apply(input, kwargs) as Tensor);
          if (recurrentDropout !== 0.0 && training) {
            expect(tfc.dropout).toHaveBeenCalledTimes(4 * (i + 1));
          } else {
            expect(tfc.dropout).toHaveBeenCalledTimes(0);
          }
          if (i === 0) {
            numTensors = tfc.memory().numTensors;
          } else {
            expect(tfc.memory().numTensors).toEqual(numTensors);
          }
        }
      });
    }
  }

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
          const input = tfc.ones([batchSize, timeSteps, inputSize]);
          let output = lstm.apply(input);

          // See comments at the beginning of this describe() block on how
          // these golden expected values can be obtained.
          const goldenOutputElementValueAtT0 = 0.7595095;
          const goldenOutputElementValueAtT1 = 0.96367633;
          const goldenHStateElementValue = goldenOutputElementValueAtT1;
          const goldenCStateElementValue = 1.99505234;

          let expectedOutput: Tensor;
          if (returnSequences) {
            const outputAtT0 = tfc.mul(
                scalar(goldenOutputElementValueAtT0),
                tfc.ones([1, batchSize, units]));
            const outputAtT1 = tfc.mul(
                scalar(goldenOutputElementValueAtT1),
                tfc.ones([1, batchSize, units]));
            expectedOutput = tfc.transpose(
                K.concatAlongFirstAxis(outputAtT0, outputAtT1), [1, 0, 2]);
          } else {
            expectedOutput = tfc.mul(
                scalar(goldenOutputElementValueAtT1),
                tfc.ones([batchSize, units]));
          }
          if (returnState) {
            output = output as Tensor[];
            expect(output.length).toEqual(3);
            expectTensorsClose(output[0], expectedOutput);
            expectTensorsClose(
                output[1],
                tfc.mul(
                    scalar(goldenHStateElementValue),
                    tfc.ones([batchSize, units])));
            expectTensorsClose(
                output[2],
                tfc.mul(
                    scalar(goldenCStateElementValue),
                    tfc.ones([batchSize, units])));
          } else {
            output = output as Tensor;
            expectTensorsClose(output, expectedOutput);
          }
        });
      }
    }

    // The reference values below can be obtained with PyKeras code:
    // ```python
    // import keras
    // import numpy as np
    //
    // units = 5
    // batch_size = 4
    // input_size = 2
    // sequence_length = 3
    //
    // rnn = keras.layers.LSTM(units,
    //                         kernel_initializer='ones',
    //                         recurrent_initializer='ones',
    //                         bias_initializer='ones',
    //                         stateful=True,
    //                         batch_input_shape=[batch_size,
    //                                            sequence_length,
    //                                            input_size])
    //
    // x = np.ones([batch_size, sequence_length, input_size])
    //
    // model = keras.Sequential()
    // model.add(rnn)
    // print(model.predict(x))
    // print(model.predict(x))
    // model.reset_states()
    // print(model.predict(x))
    // print(model.predict(x))
    // ```
    it('stateful forward', async () => {
      const sequenceLength = 3;
      const rnn = tfl.layers.lstm({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        stateful: true,
        batchInputShape: [batchSize, sequenceLength, inputSize]
      });
      const model = tfl.sequential();
      model.add(rnn);
      const x = tfc.ones([batchSize, sequenceLength, inputSize]);

      let y1 = model.predict(x) as Tensor;
      expect(y1.kept).toBe(false);
      let y1Expected =
          tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.995)));
      expectTensorsClose(y1, y1Expected);
      let y2 = model.predict(x) as Tensor;
      expect(y2.kept).toBe(false);
      // Future predicts should not dispose previous outputs.
      expect(y1.isDisposed).toBe(false);
      let y2Expected = tfc.tidy(
          () => tfc.ones([batchSize, units]).mul(tfc.scalar(0.99998766)));
      expectTensorsClose(y2, y2Expected);

      tfc.dispose([y1, y2, y1Expected, y2Expected]);
      model.resetStates();
      const numTensors0 = tfc.memory().numTensors;

      y1 = model.predict(x) as Tensor;
      expect(y1.kept).toBe(false);
      y1Expected =
          tfc.tidy(() => tfc.ones([batchSize, units]).mul(tfc.scalar(0.995)));
      expectTensorsClose(y1, y1Expected);
      y2 = model.predict(x) as Tensor;
      expect(y2.kept).toBe(false);
      // Future predicts should not dispose previous outputs.
      expect(y1.isDisposed).toBe(false);
      y2Expected = tfc.tidy(
          () => tfc.ones([batchSize, units]).mul(tfc.scalar(0.99998766)));
      expectTensorsClose(y2, y2Expected);
      tfc.dispose([y1, y2, y1Expected, y2Expected]);
      // Assert no memory leak, even without resetStates() being called.
      expect(tfc.memory().numTensors).toEqual(numTensors0);
    });

    // The reference values can be obtained with the following PyKeras code:
    // ```python
    // import keras
    // import numpy as np
    //
    // batch_size = 4
    // sequence_length = 3
    // input_size = 2
    //
    // model = keras.Sequential()
    // model.add(keras.layers.LSTM(
    //     5,
    //     batch_input_shape=[batch_size, sequence_length, input_size],
    //     kernel_initializer='ones',
    //     recurrent_initializer='ones',
    //     bias_initializer='ones',
    //     stateful=True))
    // model.add(keras.layers.Dense(1,
    //                              kernel_initializer='ones',
    //                              use_bias=False))
    // model.compile(loss='mean_squared_error', optimizer='sgd')
    //
    // xs_1 = np.ones([batch_size, sequence_length, input_size])
    // xs_2 = np.ones([batch_size, sequence_length, input_size])
    // xs = np.concatenate([xs_1, xs_2], 0)
    // ys = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    //
    // history = model.fit(xs, ys, batch_size=batch_size, shuffle=False)
    // print(history.history)
    // ```
    it('stateful BPTT', async () => {
      const sequenceLength = 3;
      const model = tfl.sequential();
      model.add(tfl.layers.lstm({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        biasInitializer: 'ones',
        stateful: true,
        batchInputShape: [batchSize, sequenceLength, inputSize]
      }));
      model.add(tfl.layers.dense(
          {units: 1, kernelInitializer: 'ones', useBias: false}));
      model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

      const xs1 = tfc.ones([batchSize, sequenceLength, inputSize]);
      const xs2 = tfc.ones([batchSize, sequenceLength, inputSize]);
      const xs = tfc.concat([xs1, xs2], 0);
      const ys = tfc.tensor2d([[1], [2], [3], [4], [5], [6], [7], [8]]);

      const history = await model.fit(xs, ys, {batchSize, shuffle: false});
      // See code snippet above for the code used to obtain this loss value.
      expect(history.history.loss[0]).toBeCloseTo(5.8377);
    });

    it('BPTT', async () => {
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
      const model = tfl.sequential();
      model.add(tfl.layers.lstm({
        units: 1,
        kernelInitializer: 'zeros',
        recurrentInitializer: 'zeros',
        useBias: false,
        inputShape: [sequenceLength, inputSize]
      }));
      model.add(tfl.layers.dense({
        units: 1,
        kernelInitializer: 'ones',
        useBias: false,
      }));

      const sgd = tfc.train.sgd(1);
      model.compile({loss: 'meanSquaredError', optimizer: sgd});

      const x = tfc.ones([batchSize, sequenceLength, inputSize]);
      const y = tfc.ones([batchSize, 1]);
      await model.fit(x, y, {epochs: 2});

      expectTensorsClose(
          model.layers[0].getWeights()[0],
          K.tile(
              tensor2d(
                  [[0.11455188, 0.06545822, 0.8760446, 0.18237013]], [1, 4]),
              [4, 1]));
      expectTensorsClose(
          model.layers[0].getWeights()[1],
          tensor2d([[0.02831176, 0.01934617, 0.00025817, 0.05784169]], [1, 4]));
      expectTensorsClose(
          model.layers[1].getWeights()[0], tensor2d([[1.4559253]], [1, 1]));
    });
  }

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Embedding(10,
  //                                  4,
  //                                  input_length=6,
  //                                  mask_zero=True,
  //                                  embeddings_initializer='ones'))
  // model.add(keras.layers.LSTM(3,
  //                             recurrent_initializer='ones',
  //                             kernel_initializer='ones',
  //                             bias_initializer='zeros'))
  // model.add(keras.layers.Dense(1,
  //                              kernel_initializer='ones',
  //                              bias_initializer='zero'))
  //
  // xs = np.array([[0, 0, 0, 0, 0, 0],
  //                [1, 0, 0, 0, 0, 0],
  //                [1, 2, 0, 0, 0, 0],
  //                [1, 2, 3, 0, 0, 0]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('With mask', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.embedding({
      inputDim: 10,
      outputDim: 4,
      inputLength: 6,
      maskZero: true,
      embeddingsInitializer: 'ones'
    }));
    model.add(tfl.layers.lstm({
      units: 3,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'}));

    const xs = tensor2d([
      [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
      [1, 2, 3, 0, 0, 0]
    ]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(
        ys, tensor2d([[0], [2.283937], [2.891939], [2.9851441]]));
  });

  // Reference Python code:
  // ```py
  // import keras
  //
  // model = keras.Sequential()
  // embedding_layer = keras.layers.Embedding(4,
  //                                          2,
  //                                          input_length=3,
  //                                          mask_zero=True)
  // model.add(embedding_layer)
  // lstm_layer = keras.layers.LSTM(2, go_backwards=True)
  // model.add(lstm_layer)
  // model.add(keras.layers.Dense(1,
  //                              kernel_initializer='ones',
  //                              bias_initializer='zero'))
  //
  // embedding_layer.set_weights([
  //     np.array([[0.1, 0.2], [0.3, 0.4], [-0.1, -0.2], [-0.3, -0.4]])])
  // print(lstm_layer.get_weights())
  // lstm_layer.set_weights([
  //     np.array([[1, 2, 3, 4, 5, 6, 7, 8],
  //               [-1, -2, -3, -4, -5, -6, -7, -8]]),
  //     np.array([[1, 2, 3, 4, 5, 6, 7, 8],
  //               [-1, -2, -3, -4, -5, -6, -7, -8]]),
  //     np.array([1, 2, 3, 4, 5, 6, 7, 8])])
  //
  // xs = np.array([[0, 0, 0],
  //                [1, 0, 0],
  //                [1, 2, 0],
  //                [1, 2, 3]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('With mask, goBackwards = true', () => {
    const model = tfl.sequential();
    const embeddingLayer = tfl.layers.embedding(
        {inputDim: 4, outputDim: 2, inputLength: 3, maskZero: true});
    model.add(embeddingLayer);
    const lstmLayer = tfl.layers.lstm({units: 2, goBackwards: true});
    model.add(lstmLayer);
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'}));

    // Setting weights to asymmetric, so that the effect of goBackwards=true
    // can show.
    embeddingLayer.setWeights(
        [tensor2d([[0.1, 0.2], [0.3, 0.4], [-0.1, -0.2], [-0.3, -0.4]])]);
    lstmLayer.setWeights([
      tensor2d([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]),
      tensor2d([[1, 2, 3, 4, 5, 6, 7, 8], [-1, -2, -3, -4, -5, -6, -7, -8]]),
      tensor1d([1, 2, 3, 4, 5, 6, 7, 8])
    ]);

    const xs = tensor2d([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(
        ys, tensor2d([[0], [1.2876499], [1.8165315], [1.9599037]]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Reshape(target_shape=[6], input_shape=[6]))
  // nested_model = keras.Sequential()
  // nested_model.add(keras.layers.Embedding(10,
  //                                         4,
  //                                         input_length=6,
  //                                         mask_zero=True,
  //                                         embeddings_initializer='ones'))
  // nested_model.add(keras.layers.LSTM(3,
  //                                    recurrent_initializer='ones',
  //                                    kernel_initializer='ones',
  //                                    bias_initializer='zeros'))
  // model.add(nested_model)
  // model.add(keras.layers.Dense(1,
  //                              kernel_initializer='ones',
  //                              bias_initializer='zero'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs = np.array([[0, 0, 0, 0, 0, 0],
  //               [1, 0, 0, 0, 0, 0],
  //               [1, 2, 0, 0, 0, 0],
  //               [1, 2, 3, 0, 0, 0]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('With mask and a nested model', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.reshape(
        {targetShape: [6], inputShape: [6]}));  // A dummy input layer.
    const nestedModel = tfl.sequential();
    nestedModel.add(tfl.layers.embedding({
      inputDim: 10,
      outputDim: 4,
      inputLength: 6,
      maskZero: true,
      embeddingsInitializer: 'ones'
    }));
    nestedModel.add(tfl.layers.lstm({
      units: 3,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.add(nestedModel);
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'}));

    const xs = tensor2d([
      [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
      [1, 2, 3, 0, 0, 0]
    ]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(
        ys, tensor2d([[0], [2.283937], [2.891939], [2.9851441]]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Embedding(10,
  //                                  4,
  //                                  input_length=6,
  //                                  mask_zero=True,
  //                                 embeddings_initializer='ones'))
  // model.add(keras.layers.LSTM(3,
  //                             recurrent_initializer='ones',
  //                             kernel_initializer='ones',
  //                             bias_initializer='zeros'))
  // model.add(keras.layers.Dense(1,
  //                             kernel_initializer='ones',
  //                             bias_initializer='zero'))
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // xs = np.array([[0, 0, 0, 0, 0, 0],
  //               [1, 0, 0, 0, 0, 0],
  //               [1, 2, 0, 0, 0, 0],
  //               [1, 2, 3, 0, 0, 0]])
  // ys = np.array([[1], [2], [3], [4]])
  //
  // model.fit(xs, ys, epochs=2, batch_size=4)
  // history = model.fit(xs, ys, epochs=2, batch_size=4)
  // print(history.history)
  // ```
  it('BPTT with mask: correctness and no leak', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.embedding({
      inputDim: 10,
      outputDim: 4,
      inputLength: 6,
      maskZero: true,
      embeddingsInitializer: 'ones'
    }));
    model.add(tfl.layers.lstm({
      units: 3,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'}));
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs = tensor2d([
      [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
      [1, 2, 3, 0, 0, 0]
    ]);
    const ys = tensor2d([[1], [2], [3], [4]]);

    // Serves as burn-in call for subsequent tracking of memory leak.
    await model.fit(xs, ys, {epochs: 2, batchSize: 4});

    const numTensors0 = tfc.memory().numTensors;
    const history = await model.fit(xs, ys, {epochs: 2, batchSize: 4});
    const numTensors1 = tfc.memory().numTensors;
    // Assert no memory leak.
    expect(numTensors1).toEqual(numTensors0);
    expect(history.history.loss.length).toEqual(2);
    expect(history.history.loss[0]).toBeCloseTo(0.503677);
    expect(history.history.loss[1]).toBeCloseTo(0.492173);
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // inp = keras.Input(shape=[6])
  // y = keras.layers.Embedding(10,
  //                            4,
  //                            input_length=6,
  //                            mask_zero=True,
  //                            embeddings_initializer='ones')(inp)
  // y = keras.layers.LSTM(3,
  //                       return_state=True,
  //                       recurrent_initializer='ones',
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y)
  //
  // model = keras.Model(inputs=inp, outputs=y)
  //
  // xs = np.array([[0, 0, 0, 0, 0, 0],
  //                [1, 0, 0, 0, 0, 0],
  //                [1, 2, 0, 0, 0, 0],
  //                [1, 2, 3, 0, 0, 0]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('With mask, returnStates = true', () => {
    const inp = tfl.input({shape: [6]});
    let y: tfl.SymbolicTensor|tfl.SymbolicTensor[] =
        tfl.layers
            .embedding({
              inputDim: 10,
              outputDim: 4,
              inputLength: 6,
              maskZero: true,
              embeddingsInitializer: 'ones'
            })
            .apply(inp) as tfl.SymbolicTensor;
    y = tfl.layers
            .lstm({
              units: 3,
              returnState: true,
              recurrentInitializer: 'ones',
              kernelInitializer: 'ones',
              biasInitializer: 'zeros'
            })
            .apply(y) as tfl.SymbolicTensor[];
    const model = tfl.model({inputs: inp, outputs: y});
    const xs = tensor2d([
      [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0],
      [1, 2, 3, 0, 0, 0]
    ]);
    const ys = model.predict(xs) as Tensor[];
    expect(ys.length).toEqual(3);
    expectTensorsClose(ys[0], tensor2d([
                         [0, 0, 0], [0.76131237, 0.76131237, 0.76131237],
                         [0.9639796, 0.9639796, 0.9639796],
                         [0.99504817, 0.99504817, 0.99504817]
                       ]));
    expectTensorsClose(ys[1], tensor2d([
                         [0, 0, 0], [0.76131237, 0.76131237, 0.76131237],
                         [0.9639796, 0.9639796, 0.9639796],
                         [0.99504817, 0.99504817, 0.99504817]
                       ]));
    expectTensorsClose(
        ys[2], tensor2d([
          [0, 0, 0], [0.9993292, 0.9993292, 0.9993292],
          [1.9993222, 1.9993222, 1.9993222], [2.9993203, 2.9993203,　2.9993203]
        ]));
  });

  // Referernce Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Embedding(10,
  //                                 4,
  //                                 input_length=3,
  //                                 mask_zero=True,
  //                                 embeddings_initializer='ones'))
  // model.add(keras.layers.LSTM(3,
  //                             return_sequences=True,
  //                             recurrent_initializer='ones',
  //                             kernel_initializer='ones',
  //                             bias_initializer='zeros'))
  // model.add(keras.layers.Dense(1,
  //                             kernel_initializer='ones',
  //                             bias_initializer='zeros'))
  //
  // xs = np.array([[0, 0, 0],
  //               [1, 0, 0],
  //               [1, 2, 0],
  //               [1, 2, 3]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('With mask, returnSequences = true', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.embedding({
      inputDim: 10,
      outputDim: 4,
      inputLength: 3,
      maskZero: true,
      embeddingsInitializer: 'ones'
    }));
    model.add(tfl.layers.lstm({
      units: 3,
      returnSequences: true,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.add(tfl.layers.dense(
        {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'}));

    const xs = tensor2d([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]]);
    const ys = model.predict(xs) as Tensor;
    expectTensorsClose(ys, tensor3d([
                         [[0], [0], [0]], [[2.283937], [2.283937], [2.283937]],
                         [[2.283937], [2.8919387], [2.8919387]],
                         [[2.283937], [2.8919387], [2.9851446]]
                       ]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // model = keras.Sequential()
  // model.add(keras.layers.Embedding(10,
  //                                  4,
  //                                  input_length=3,
  //                                  mask_zero=True,
  //                                  embeddings_initializer='ones'))
  // model.add(keras.layers.SimpleRNN(3,
  //                                  return_sequences=True,
  //                                  recurrent_initializer='ones',
  //                                  kernel_initializer='ones',
  //                                  bias_initializer='zeros'))
  // model.add(keras.layers.LSTM(3,
  //                             recurrent_initializer='ones',
  //                             kernel_initializer='ones',
  //                             bias_initializer='zeros'))
  //
  // xs = np.array([[0, 0, 0],
  //                [1, 0, 0],
  //                [1, 2, 0],
  //                [1, 2, 3]])
  // ys = model.predict(xs)
  // print(ys)
  // ```
  it('Stacked RNNs with masking: correctness and no leak', () => {
    const model = tfl.sequential();
    model.add(tfl.layers.embedding({
      inputDim: 10,
      outputDim: 4,
      inputLength: 3,
      maskZero: true,
      embeddingsInitializer: 'ones'
    }));
    model.add(tfl.layers.simpleRNN({
      units: 3,
      returnSequences: true,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));
    model.add(tfl.layers.lstm({
      units: 3,
      recurrentInitializer: 'ones',
      kernelInitializer: 'ones',
      biasInitializer: 'zeros'
    }));

    const xs = tensor2d([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]]);

    // Burn-in call for subsequent memory leak check.
    model.predict(xs);

    const numTensors0 = tfc.memory().numTensors;
    const ys = model.predict(xs) as Tensor;
    const numTensors1 = tfc.memory().numTensors;
    expectTensorsClose(ys, tensor2d([
                         [0, 0, 0], [0.75950104, 0.75950104, 0.75950104],
                         [0.96367145, 0.96367145, 0.96367145],
                         [0.9950049, 0.9950049, 0.9950049]
                       ]));
    ys.dispose();
    // Assert no memory leak.
    expect(numTensors1).toEqual(numTensors0 + 1);
  });
});

describeMathCPU('LSTM-deserialization', () => {
  it('modelFromConfig', async () => {
    const model = await modelFromJSON(fakeLSTMModel);
    const encoderInputs = tfc.zeros([1, 3, 71], 'float32');
    const decoderInputs = tfc.zeros([1, 3, 94], 'float32');
    const outputs = model.predict([encoderInputs, decoderInputs]) as Tensor;
    expect(outputs.shape).toEqual([1, 3, 94]);
  });

  it('Default recurrentActivation round trip', () => {
    const x = randomNormal([1, 2, 3]);
    const layer = tfl.layers.lstm({units: 4});
    const y = layer.apply(x) as Tensor;
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.lstm(tsConfig);
    const yPrime = layer.apply(x) as Tensor;
    expectTensorsClose(yPrime, y);
    expect(layerPrime.getConfig()['recurrentActivation'])
        .toEqual(layer.getConfig()['recurrentActivation']);
  });

  it('Non-default recurrentActivation round trip', () => {
    const x = randomNormal([1, 2, 3]);
    const layer =
        tfl.layers.lstm({units: 4, recurrentActivation: 'tanh'});
    const y = layer.apply(x) as Tensor;
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.lstm(tsConfig);
    const yPrime = layer.apply(x) as Tensor;
    expectTensorsClose(yPrime, y);
    expect(layerPrime.getConfig()['recurrentActivation'])
        .toEqual(layer.getConfig()['recurrentActivation']);
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
          'inbound_nodes': [
            [['input_2', 0, 0, {}], ['lstm_1', 0, 1, {}], ['lstm_1', 0, 2, {}]]
          ],
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

describeMathCPU('LSTM Serialization', () => {
  const cellConfig: LSTMCellLayerArgs = {
    units: 8,
    activation: 'tanh',
    recurrentActivation: 'tanh',
    useBias: true,
    kernelInitializer: 'glorotUniform',
    recurrentInitializer: 'heUniform',
    biasInitializer: 'ones',
    kernelRegularizer: 'l1l2',
    recurrentRegularizer: 'l1l2',
    biasRegularizer: 'l1l2',
    kernelConstraint: 'unitNorm',
    recurrentConstraint: 'unitNorm',
    biasConstraint: 'nonNeg',
    dropout: 0.1,
    recurrentDropout: 0.2,
    name: 'cell_1',
    batchSize: 12,
    batchInputShape: [12, 8, 8],
    inputShape: [8, 8],
    dtype: 'int32',
    inputDType: 'int32',
    trainable: true,
    implementation: 1,
    unitForgetBias: true,
  };

  const expectedCellConfigPrime = {
    name: 'cell_1',
    trainable: true,
    batchInputShape: [12, 8, 8],
    dtype: 'int32',
    units: 8,
    activation: serializeActivation(new Tanh()),
    recurrentActivation: serializeActivation(new Tanh()),
    useBias: true,
    kernelInitializer: serializeInitializer(new GlorotUniform()),
    recurrentInitializer: serializeInitializer(new HeUniform()),
    biasInitializer: serializeInitializer(new Ones()),
    kernelRegularizer: serializeRegularizer(new L1L2()),
    recurrentRegularizer: serializeRegularizer(new L1L2()),
    biasRegularizer: serializeRegularizer(new L1L2()),
    activityRegularizer: serializeRegularizer(null),
    kernelConstraint: serializeConstraint(new UnitNorm({})),
    recurrentConstraint: serializeConstraint(new UnitNorm({})),
    biasConstraint: serializeConstraint(new NonNeg()),
    implementation: 1,
    unitForgetBias: true,
  };

  describe('LSTMCell.getConfig', () => {
    it('should return the expected values', () => {
      const cell = tfl.layers.lstmCell(cellConfig);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual(expectedCellConfigPrime);
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });

  describe('LSTM.getConfig', () => {
    it('should return the expected values', () => {
      const config: LSTMLayerArgs = {
        ...cellConfig,
        name: 'layer_1',
        ...{
          returnSequences: true,
          returnState: true,
          stateful: true,
          unroll: true,
          goBackwards: true,
          inputDim: 8,
          inputLength: 8,
        } as Omit<LSTMLayerArgs, keyof LSTMCellLayerArgs>
      };

      const cell = tfl.layers.lstm(config);

      const {dropout, recurrentDropout, ...configPrime} = cell.getConfig();

      expect(configPrime).toEqual({
        ...expectedCellConfigPrime,
        name: 'layer_1',
        returnSequences: true,
        returnState: true,
        stateful: true,
        unroll: true,
        goBackwards: true,
      });
      expect(dropout).toBeCloseTo(0.1);
      expect(recurrentDropout).toBeCloseTo(0.2);
    });
  });
});

describeMathCPU('StackedRNNCells Symbolic', () => {
  it('With SimpleRNNCell', () => {
    const stackedRNN = tfl.layers.rnn({
      cell: tfl.layers.stackedRNNCells({
        cells: [
          tfl.layers.simpleRNNCell(
              {units: 3, recurrentInitializer: 'glorotNormal'}),
          tfl.layers.simpleRNNCell(
              {units: 2, recurrentInitializer: 'glorotNormal'})
        ],
      })
    });
    const input =
        new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
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
        new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
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
        new tfl.SymbolicTensor('float32', [16, 10, 7], null, [], null);
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

describeMathCPU('Stacked RNN serialization', () => {
  it('StackedRNNCells', async () => {
    const model = tfl.sequential();
    model.add(tfl.layers.dense(
        {units: 1, inputShape: [3, 4], kernelInitializer: 'ones'}));
    const cells = [
      tfl.layers.lstmCell(
          {units: 5, kernelInitializer: 'ones', recurrentInitializer: 'ones'}),
      tfl.layers.lstmCell(
          {units: 6, kernelInitializer: 'ones', recurrentInitializer: 'ones'})
    ];
    const rnn = tfl.layers.rnn({cell: cells, returnSequences: true});
    model.add(rnn);
    const xs = tfc.ones([1, 3, 4]).mul(0.1);
    const ys = model.predict(xs) as Tensor;

    const modelJSON = model.toJSON(null, false);
    const modelPrime =
        await tfl.models.modelFromJSON({modelTopology: modelJSON});
    const ysPrime = modelPrime.predict(xs) as Tensor;
    expectTensorsClose(ysPrime, ys);
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
