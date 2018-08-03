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
 * Unit tests for wrapper layers.
 */

import {ones, scalar, serialization, Tensor, tensor2d, Tensor3D, tensor3d} from '@tensorflow/tfjs-core';

import {Layer, SymbolicTensor} from '../engine/topology';
import {Model} from '../engine/training';
import * as tfl from '../index';
import {convertPythonicToTs} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Dense, Reshape} from './core';
import {RNN, SimpleRNN} from './recurrent';
import {deserialize} from './serialization';
import {Bidirectional, BidirectionalMergeMode, checkBidirectionalMergeMode, TimeDistributed, VALID_BIDIRECTIONAL_MERGE_MODES} from './wrappers';


describeMathCPU('TimeDistributed Layer: Symbolic', () => {
  it('3D input: Dense', () => {
    const input = new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    const output = wrapper.apply(input) as tfl.SymbolicTensor;
    expect(wrapper.trainable).toEqual(true);
    expect(wrapper.getWeights().length).toEqual(2);  // kernel and bias.
    expect(output.dtype).toEqual(input.dtype);
    expect(output.shape).toEqual([10, 8, 3]);
  });
  it('4D input: Reshape', () => {
    const input =
        new tfl.SymbolicTensor('float32', [10, 8, 2, 3], null, [], null);
    const wrapper =
        tfl.layers.timeDistributed({layer: new Reshape({targetShape: [6]})});
    const output = wrapper.apply(input) as tfl.SymbolicTensor;
    expect(output.dtype).toEqual(input.dtype);
    expect(output.shape).toEqual([10, 8, 6]);
  });
  it('2D input leads to exception', () => {
    const input = new tfl.SymbolicTensor('float32', [10, 2], null, [], null);
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    expect(() => wrapper.apply(input))
        .toThrowError(
            /TimeDistributed .*expects an input shape >= 3D, .* \[10,.*2\]/);
  });
  it('getConfig and fromConfig: round trip', () => {
    const wrapper = tfl.layers.timeDistributed({layer: new Dense({units: 3})});
    const config = wrapper.getConfig();
    const wrapperPrime = TimeDistributed.fromConfig(TimeDistributed, config);
    expect(wrapperPrime.getConfig()).toEqual(wrapper.getConfig());
  });
});

describeMathCPUAndGPU('TimeDistributed Layer: Tensor', () => {
  it('3D input: Dense', () => {
    const input = tensor3d(
        [
          [[1, 2], [3, 4], [5, 6], [7, 8]],
          [[-1, -2], [-3, -4], [-5, -6], [-7, -8]]
        ],
        [2, 4, 2]);
    // Given an all-ones Dense kernel and no bias, the output at each timestep
    // is expected to be [3, 7, 11, 15], give or take a minus sign.
    const wrapper = tfl.layers.timeDistributed({
      layer: new Dense({units: 1, kernelInitializer: 'ones', useBias: false})
    });
    const output = wrapper.apply(input) as Tensor;
    expectTensorsClose(
        output,
        tensor3d(
            [[[3], [7], [11], [15]], [[-3], [-7], [-11], [-15]]], [2, 4, 1]));
  });
});

describeMathCPU('Bidirectional Layer: Symbolic', () => {
  const mergeModes: BidirectionalMergeMode[] = [
    null,
    'concat',
    'ave',
    'mul',
    'sum',
  ];
  const returnStateValues: boolean[] = [false, true];

  for (const mergeMode of mergeModes) {
    for (const returnState of returnStateValues) {
      const testTitle = `3D input: returnSequence=false, ` +
          `mergeMode=${mergeMode}; returnState=${returnState}`;
      it(testTitle, () => {
        const input =
            new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
        const bidi = tfl.layers.bidirectional({
          layer: new SimpleRNN(
              {units: 3, recurrentInitializer: 'glorotNormal', returnState}),
          mergeMode,
        });
        // TODO(cais): Remove recurrentInitializer once Orthogonal initializer
        // is available.
        let outputs = bidi.apply(input);
        expect(bidi.trainable).toEqual(true);
        // {kernel, recurrentKernel, bias} * {forward, backward}.
        expect(bidi.getWeights().length).toEqual(6);
        if (!returnState) {
          if (mergeMode === null) {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(2);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
          } else if (mergeMode === 'concat') {
            outputs = outputs as tfl.SymbolicTensor;
            expect(outputs.shape).toEqual([10, 6]);
          } else {
            outputs = outputs as tfl.SymbolicTensor;
            expect(outputs.shape).toEqual([10, 3]);
          }
        } else {
          if (mergeMode === null) {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(4);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
            expect(outputs[3].shape).toEqual([10, 3]);
          } else if (mergeMode === 'concat') {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(3);
            expect(outputs[0].shape).toEqual([10, 6]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
          } else {
            outputs = outputs as tfl.SymbolicTensor[];
            expect(outputs.length).toEqual(3);
            expect(outputs[0].shape).toEqual([10, 3]);
            expect(outputs[1].shape).toEqual([10, 3]);
            expect(outputs[2].shape).toEqual([10, 3]);
          }
        }
      });
    }
  }
  it('returnSequence=true', () => {
    const input = new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
    const bidi = tfl.layers.bidirectional({
      layer: new SimpleRNN({
        units: 3,
        recurrentInitializer: 'glorotNormal',
        returnSequences: true,
        returnState: true
      }),
      mergeMode: 'ave'
    });
    const outputs = bidi.apply(input) as tfl.SymbolicTensor[];
    expect(outputs.length).toEqual(3);
    expect(outputs[0].shape).toEqual([10, 8, 3]);
    expect(outputs[1].shape).toEqual([10, 3]);
    expect(outputs[2].shape).toEqual([10, 3]);
  });
  it('Serialization round trip', () => {
    const layer = tfl.layers.bidirectional({
      layer: new SimpleRNN({units: 3}),
      mergeMode: 'concat',
      inputShape: [4, 4],
    });
    const model = tfl.sequential({layers: [layer]});
    const unused: {} = null;
    const modelJSON = model.toJSON(unused, false);
    const modelPrime = deserialize(
                           convertPythonicToTs(modelJSON) as
                           serialization.ConfigDict) as tfl.Sequential;
    expect((modelPrime.layers[0] as Bidirectional).getConfig().mergeMode)
        .toEqual('concat');
  });
});

describe('checkBidirectionalMergeMode', () => {
  it('Valid values', () => {
    const extendedValues =
        VALID_BIDIRECTIONAL_MERGE_MODES.concat([undefined, null]);
    for (const validValue of extendedValues) {
      // Using implicit "expect().toNotThrow()" for valid values
      checkBidirectionalMergeMode(validValue);
    }
  });
  it('Invalid values', () => {
    // Test invalid values are rejected, and reported in the error.
    expect(() => checkBidirectionalMergeMode('foo')).toThrowError(/foo/);
    try {
      checkBidirectionalMergeMode('bad');
    } catch (e) {
      expect(e).toMatch('BidirectionalMergeMode');
      // Test that the error message contains the list of valid values.
      for (const validValue of VALID_BIDIRECTIONAL_MERGE_MODES) {
        expect(e).toMatch(validValue);
      }
    }
  });
});

describeMathCPUAndGPU('Bidirectional Layer: Tensor', () => {
  // The golden tensor values used in the tests below can be obtained with
  // PyKeras code such as the following:
  // ```python
  // import keras
  // import numpy as np
  //
  // bidi = keras.layers.Bidirectional(
  //     keras.layers.SimpleRNN(
  //         3,
  //         kernel_initializer='ones',
  //         recurrent_initializer='ones',
  //         return_state=True),
  //     merge_mode='ave')
  //
  // time_steps = 4
  // input_size = 2
  // inputs = keras.Input([time_steps, input_size])
  // outputs = bidi(inputs)
  // model = keras.Model(inputs, outputs)
  //
  // x = np.array(
  //     [[[0.05, 0.05],
  //       [-0.05, -0.05],
  //       [0.1, 0.1],
  //       [-0.1, -0.1]]])
  // print(model.predict(x))
  // ```

  // TODO(bileschi): This should be tfl.layers.Layer.
  let bidi: Layer;
  let x: Tensor3D;
  function createLayerAndData(
      mergeMode: BidirectionalMergeMode, returnState: boolean) {
    const units = 3;
    bidi = tfl.layers.bidirectional({
      layer: new SimpleRNN({
        units,
        kernelInitializer: 'ones',
        recurrentInitializer: 'ones',
        useBias: false,
        returnState
      }),
      mergeMode,
    });
    const timeSteps = 4;
    const inputSize = 2;
    x = tensor3d(
        [[[0.05, 0.05], [-0.05, -0.05], [0.1, 0.1], [-0.1, -0.1]]],
        [1, timeSteps, inputSize]);
  }

  const mergeModes: BidirectionalMergeMode[] = [null, 'concat', 'mul'];
  for (const mergeMode of mergeModes) {
    it(`No returnState, mergeMode=${mergeMode}`, () => {
      createLayerAndData(mergeMode, false);
      let y = bidi.apply(x);
      if (mergeMode === null) {
        y = y as Tensor[];
        expect(y.length).toEqual(2);
        expectTensorsClose(
            y[0], tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
        expectTensorsClose(
            y[1], tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
      } else if (mergeMode === 'concat') {
        y = y as Tensor;
        expectTensorsClose(
            y,
            tensor2d(
                [[
                  0.9440416, 0.9440416, 0.9440416, -0.9842659, -0.9842659,
                  -0.9842659
                ]],
                [1, 6]));
      } else if (mergeMode === 'mul') {
        y = y as Tensor;
        expectTensorsClose(
            y, tensor2d([[-0.929188, -0.929188, -0.929188]], [1, 3]));
      }
    });
  }
  it('returnState', () => {
    createLayerAndData('ave', true);
    const y = bidi.apply(x) as Tensor[];
    expect(y.length).toEqual(3);
    expectTensorsClose(
        y[0], tensor2d([[-0.02011216, -0.02011216, -0.02011216]], [1, 3]));
    expectTensorsClose(
        y[1], tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
    expectTensorsClose(
        y[2], tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
  });

  // The golden values in the test below can be obtained with the following
  // Python Keras code.
  //
  // ```python
  // import keras
  // import numpy
  //
  // rnn = keras.layers.LSTM(
  //     1,
  //     kernel_initializer='ones',
  //     recurrent_initializer='ones',
  //     bias_initializer='ones',
  //     go_backwards=True)
  // bidi = keras.layers.Bidirectional(
  //     rnn, merge_mode='concat', input_shape=[2, 2])
  // model = keras.Sequential([bidi])
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // x = np.array([[[0.1, 0.2],
  //               [-0.1, 0.1]]])
  // y = np.array([[0.3, 0.5]])
  // print(model.predict(x))
  //
  // history = model.fit(x, y)
  // print(history.history)
  // ```
  it('Backwards LSTM: predict and fit: concat', async () => {
    const lstm = tfl.layers.lstm({
      units: 1,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'ones',
      goBackwards: true
    }) as RNN;
    const bidi = tfl.layers.bidirectional(
        {layer: lstm, inputShape: [2, 2], mergeMode: 'concat'});
    const model = tfl.sequential({layers: [bidi]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const x = tensor3d([[[0.1, 0.2], [-0.1, 0.1]]]);
    const y = tensor2d([[0.3, 0.5]]);
    expectTensorsClose(
        model.predict(x) as Tensor, tensor2d([[0.69299805, 0.66088]]));
    const history = await model.fit(x, y);
    expect(history.history.loss[0]).toBeCloseTo(0.0901649);
    expectTensorsClose(
        model.predict(x) as Tensor, tensor2d([[0.6927189, 0.66077083]]));
  });

  // The golden values in the test below can be obtained with the following
  // Python Keras code.
  //
  // ```python
  // import keras
  // import numpy
  //
  // rnn = keras.layers.LSTM(
  //     1,
  //     kernel_initializer='ones',
  //     recurrent_initializer='ones',
  //     bias_initializer='ones',
  //     go_backwards=True)
  // bidi = keras.layers.Bidirectional(
  //     rnn, merge_mode='ave', input_shape=[2, 2])
  // model = keras.Sequential([bidi])
  // model.compile(loss='mean_squared_error', optimizer='sgd')
  //
  // x = np.array([[[0.1, 0.2],
  //               [-0.1, 0.1]]])
  // y = np.array([[0.4]])
  // print(model.predict(x))
  //
  // history = model.fit(x, y)
  // print(history.history)
  // ```
  it('Backwards LSTM: predict and fit: ave', done => {
    const lstm = tfl.layers.lstm({
      units: 1,
      kernelInitializer: 'ones',
      recurrentInitializer: 'ones',
      biasInitializer: 'ones',
      goBackwards: true
    }) as RNN;
    const bidi = tfl.layers.bidirectional(
        {layer: lstm, inputShape: [2, 2], mergeMode: 'ave'});
    const model = tfl.sequential({layers: [bidi]});
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const x = tensor3d([[[0.1, 0.2], [-0.1, 0.1]]]);
    const y = tensor2d([[0.4]]);
    expectTensorsClose(model.predict(x) as Tensor, tensor2d([[0.676939]]));
    model.fit(x, y)
        .then(history => {
          expect(history.history.loss[0]).toBeCloseTo(0.0766952);
          expectTensorsClose(
              model.predict(x) as Tensor, tensor2d([[0.6767467]]));
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });
});

describeMathCPUAndGPU('Bidirectional with initial state', () => {
  const sequenceLength = 4;
  const recurrentUnits = 3;
  const inputDim = 2;

  function createLayerAndTensors(): {
    bidiLayer: Bidirectional,
    initState1: SymbolicTensor,
    initState2: SymbolicTensor,
    inputTensor: SymbolicTensor,
    outputTensors: SymbolicTensor[],
  } {
    const initState1 = tfl.input({shape: [recurrentUnits]});
    const initState2 = tfl.input({shape: [recurrentUnits]});
    const inputTensor = tfl.input({shape: [sequenceLength, inputDim]});
    const bidiLayer = tfl.layers.bidirectional({
      layer: tfl.layers.gru({
        units: recurrentUnits,
        kernelInitializer: 'zeros',
        recurrentInitializer: 'zeros',
        biasInitializer: 'ones'
      }) as RNN,
      mergeMode: null,
    }) as Bidirectional;
    const outputTensors =
        bidiLayer.apply(
            inputTensor, {initialState: [initState1, initState2]}) as
        SymbolicTensor[];
    return {bidiLayer, initState1, initState2, inputTensor, outputTensors};
  }

  it('Correct shapes', () => {
    const layerAndTensors = createLayerAndTensors();
    expect(layerAndTensors.outputTensors.length).toEqual(2);
    expect(layerAndTensors.outputTensors[0].shape).toEqual([
      null, recurrentUnits
    ]);
    expect(layerAndTensors.outputTensors[1].shape).toEqual([
      null, recurrentUnits
    ]);
  });

  it('apply() with concrete tensors', () => {
    const layerAndTensors = createLayerAndTensors();
    const xVal = ones([1, sequenceLength, inputDim]);
    const initState1Val = ones([1, recurrentUnits]).mul(scalar(-1));
    const initState2Val = ones([1, recurrentUnits]);
    const yVals =
        layerAndTensors.bidiLayer.apply(
            xVal, {initialState: [initState1Val, initState2Val]}) as Tensor[];
    expect(yVals.length).toEqual(2);
    expectTensorsClose(
        yVals[0], tensor2d([[0.33863544, 0.33863544, 0.33863544]]));
    expectTensorsClose(yVals[1], tensor2d([[0.8188354, 0.8188354, 0.8188354]]));
  });

  it('Model predict', () => {
    const layerAndTensors = createLayerAndTensors();
    const model = tfl.model({
      inputs: [
        layerAndTensors.inputTensor, layerAndTensors.initState1,
        layerAndTensors.initState2
      ],
      outputs: layerAndTensors.outputTensors
    });
    const xVal = ones([1, sequenceLength, inputDim]);
    const initState1Val = ones([1, recurrentUnits]).mul(scalar(-1));
    const initState2Val = ones([1, recurrentUnits]);
    const yVals =
        model.predict([xVal, initState1Val, initState2Val]) as Tensor[];
    expect(yVals.length).toEqual(2);
    expectTensorsClose(
        yVals[0], tensor2d([[0.33863544, 0.33863544, 0.33863544]]));
    expectTensorsClose(yVals[1], tensor2d([[0.8188354, 0.8188354, 0.8188354]]));
  });

  it('Model serialization round trip', () => {
    // Disable the console warning about the unserialization SymbolicTensor in
    // the CallArgs.
    spyOn(console, 'warn');
    const layerAndTensors = createLayerAndTensors();
    const model = tfl.model({
      inputs: [
        layerAndTensors.inputTensor, layerAndTensors.initState1,
        layerAndTensors.initState2
      ],
      outputs: layerAndTensors.outputTensors
    });
    const json1 = model.toJSON(null, false);
    const model2 =
        deserialize(convertPythonicToTs(json1) as serialization.ConfigDict) as
        Model;
    const json2 = model2.toJSON(null, false);
    expect(json2).toEqual(json1);
  });

  it('Incorrect number of initial-state tensors leads to error', () => {
    const initState1 = tfl.input({shape: [recurrentUnits]});
    const x = tfl.input({shape: [sequenceLength, inputDim]});
    const bidi = tfl.layers.bidirectional({
      layer: tfl.layers.gru({
        units: recurrentUnits,
      }) as RNN,
      mergeMode: null,
    });
    expect(() => bidi.apply(x, {
      initialState: [initState1]
    })).toThrowError(/the state should be .*RNNs/);
  });
});
