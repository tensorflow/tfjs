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
 * Unit tests for core.ts.
 */

import {ones, serialization, Tensor, tensor1d, Tensor2D, tensor2d, tensor3d} from '@tensorflow/tfjs-core';

import {SymbolicTensor} from '../engine/topology';
import * as tfl from '../index';
import {Shape} from '../keras_format/common';
import {deserialize} from '../layers/serialization';
import {convertPythonicToTs, convertTsToPythonic} from '../utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Add, Average, Concatenate, Maximum, Minimum, Multiply} from './merge';

describeMathCPU('Merge Layers Except Concatenate: Symbolic', () => {
  const layers = [Add, Average, Multiply, Maximum, Minimum];
  const symbolicInputShapes: Shape[] = [
    [10, 3],
    [10, 2, 2],
  ];
  const numInputsArray: number[] = [2, 4];

  for (const layer of layers) {
    for (const inputShape of symbolicInputShapes) {
      for (const numInputs of numInputsArray) {
        const testTitle =
            `layer=${layer.name}; inputShape=${JSON.stringify(inputShape)}; ` +
            `numInputs=${numInputs}`;
        it(testTitle, () => {
          const addLayer = new layer({name: layer.name});
          const symbolicInputs: tfl.SymbolicTensor[] = [];
          for (let i = 0; i < numInputs; ++i) {
            symbolicInputs.push(
                new tfl.SymbolicTensor('float32', inputShape, null, [], null));
          }
          const output = addLayer.apply(symbolicInputs) as tfl.SymbolicTensor;
          expect(output.dtype).toEqual(symbolicInputs[0].dtype);
          expect(output.shape).toEqual(inputShape);
        });
      }
    }
  }

  it('Single input leads to exception', () => {
    const x = new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
    const addLayer = tfl.layers.add({name: 'Add'});
    expect(() => {
      addLayer.apply([x]);
    }).toThrowError(/.*at least 2 inputs\. Got 1 input.*/);
  });

  it('Non-unique batch sizes to exception', () => {
    const x1 = new tfl.SymbolicTensor('float32', [1, 2], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
    const addLayer = tfl.layers.add({name: 'Add'});
    expect(() => {
      addLayer.apply([x1, x2]);
    }).toThrowError(/Can not merge tensors with different batch sizes/);
  });
});

describeMathCPUAndGPU('Add-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect((tfl.layers.add()).getClassName()).toEqual('Add');
  });

  it('Calling with config arg returns Layer', () => {
    expect((tfl.layers.add({name: 'addLayer'})).name.indexOf('addLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 2]});
    const input2 = tfl.layers.input({shape: [2, 2]});
    const output =
        tfl.layers.add().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = tfl.layers.add().apply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([11, 22, 33, 44], [2, 2]));
  });

  it('predict() with functional model with Add layer works', () => {
    const input = tfl.layers.input({shape: [24, 24, 3]});
    const conv1 =
        tfl.layers.conv2d({filters: 4, kernelSize: [3, 3]}).apply(input) as
        tfl.SymbolicTensor;
    const conv2 =
        tfl.layers.conv2d({filters: 4, kernelSize: [3, 3]}).apply(input) as
        tfl.SymbolicTensor;
    const sum = tfl.layers.add().apply([conv1, conv2]) as tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input], outputs: sum});
    const x = ones([1, 24, 24, 3]);
    const y = model.predict(x) as Tensor;
    expect(y.shape).toEqual([1, 22, 22, 4]);
  });
});

describeMathCPUAndGPU('Multiply-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(tfl.layers.multiply().getClassName()).toEqual('Multiply');
  });

  it('Calling with config arg returns Layer', () => {
    expect(tfl.layers.multiply({name: 'multiplyLayer'})
               .name.indexOf('multiplyLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 2]});
    const input2 = tfl.layers.input({shape: [2, 2]});
    const output =
        tfl.layers.multiply().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = tfl.layers.multiply().apply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([10, 40, 90, 160], [2, 2]));
  });
});

describeMathCPUAndGPU('Average-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(tfl.layers.average().getClassName()).toEqual('Average');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        tfl.layers.average({name: 'averageLayer'}).name.indexOf('averageLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 2]});
    const input2 = tfl.layers.input({shape: [2, 2]});
    const output =
        tfl.layers.average().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = tfl.layers.average().apply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([5.5, 11, 16.5, 22], [2, 2]));
  });
});

describeMathCPUAndGPU('Maximum-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(tfl.layers.maximum().getClassName()).toEqual('Maximum');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        tfl.layers.maximum({name: 'maximumLayer'}).name.indexOf('maximumLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 2]});
    const input2 = tfl.layers.input({shape: [2, 2]});
    const output =
        tfl.layers.maximum().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 20, 3, 40], [2, 2]);
    const input2 = tensor2d([10, 2, 30, 4], [2, 2]);
    const output = tfl.layers.maximum().apply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([10, 20, 30, 40], [2, 2]));
  });
});

describeMathCPUAndGPU('Minimum-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(tfl.layers.minimum().getClassName()).toEqual('Minimum');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        tfl.layers.minimum({name: 'minimumLayer'}).name.indexOf('minimumLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 2]});
    const input2 = tfl.layers.input({shape: [2, 2]});
    const output =
        tfl.layers.minimum().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 20, 3, 40], [2, 2]);
    const input2 = tensor2d([10, 2, 30, 4], [2, 2]);
    const output = tfl.layers.minimum().apply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([1, 2, 3, 4], [2, 2]));
  });
});

describeMathCPUAndGPU('Concatenate-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(tfl.layers.concatenate().getClassName()).toEqual('Concatenate');
  });

  it('Calling with config arg returns Layer', () => {
    expect(tfl.layers.concatenate({name: 'concatenateLayer'})
               .name.indexOf('concatenateLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = tfl.layers.input({shape: [2, 3]});
    const input2 = tfl.layers.input({shape: [2, 4]});
    const output =
        tfl.layers.concatenate().apply([input1, input2]) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 7]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const input2 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const output = tfl.layers.concatenate().apply([input1, input2]) as Tensor;
    expectTensorsClose(
        output, tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]));
  });
});

describeMathCPU('Concatenate Layer: Symbolic', () => {
  it('All known shapes', () => {
    const x1 = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
    const layer0 = tfl.layers.concatenate({});
    expect((layer0.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      2, 3, 8
    ]);
    const layer1 = tfl.layers.concatenate({axis: -1});
    expect((layer1.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      2, 3, 8
    ]);
    const layer2 = tfl.layers.concatenate({axis: 0});
    expect((layer2.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      4, 3, 4
    ]);
    const layer3 = tfl.layers.concatenate({axis: 1});
    expect((layer3.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      2, 6, 4
    ]);
  });
  it('Concat axis has unknown shape', () => {
    const x1 = new tfl.SymbolicTensor('float32', [2, null, 4], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [2, null, 4], null, [], null);
    const layer = tfl.layers.concatenate({axis: 1});
    expect((layer.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      2, null, 4
    ]);
  });
  it('Non-concat axis has unknown shape', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 3, 4], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 5, 4], null, [], null);
    const layer = tfl.layers.concatenate({axis: 1});
    expect((layer.apply([x1, x2]) as tfl.SymbolicTensor).shape).toEqual([
      null, 8, 4
    ]);
  });
  it('Incompatible shape leads to error', () => {
    const x1 = new tfl.SymbolicTensor('float32', [2, 3, 5], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [2, 4, 5], null, [], null);
    const layer = tfl.layers.concatenate({});
    expect(() => layer.apply([
      x1, x2
    ])).toThrowError(/requires inputs with matching shapes except/);
  });
  it('Single shape leads to error', () => {
    const x1 = new tfl.SymbolicTensor('float32', [2, 3, 5], null, [], null);
    const layer = tfl.layers.concatenate({});
    expect(() => layer.apply([x1]))
        .toThrowError(/should be called on a list of at least 2 inputs/);
  });
  it('Serialization round trip', () => {
    const layer = tfl.layers.concatenate({axis: 2});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.concatenate(tsConfig);
    expect(layerPrime.getConfig().axis).toEqual(2);
  });
});

describeMathCPUAndGPU('Add Layer: Tensor', () => {
  it('2D plus 2D', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const addLayer = tfl.layers.add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[9, 18], [27, 36]], [2, 2]));
  });
  it('2D plus 2D, with broadcast', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const addLayer = tfl.layers.add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[8, 18], [26, 36]], [2, 2]));
  });
  it('2D plus 2D, with dimension expansion', () => {
    const x1 =
        tensor3d([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], [2, 2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const addLayer = tfl.layers.add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(
        y, tensor3d([[[8, 18], [28, 38]], [[46, 56], [66, 76]]], [2, 2, 2]));
  });
  it('computeMask', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-2, -1], [-4, -3]]);
    const addLayer = tfl.layers.add({});
    const m1 = tensor1d([true, false], 'bool');
    const m2 = tensor1d([true, true], 'bool');
    const mask = addLayer.computeMask([x1, x2], [m1, m2]);
    expectTensorsClose(mask, tensor2d([[true, false]], [1, 2], 'bool'));
  });
  it('computeMask error condition: non-array input', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-2, -1], [-4, -3]]);
    const addLayer = tfl.layers.add({});
    const m1 = tensor1d([true, false], 'bool');
    const m2 = tensor1d([true, true], 'bool');
    expect(() => addLayer.computeMask(x1, [
      m1, m2
    ])).toThrowError(/inputs.*should be an Array/);
    expect(() => addLayer.computeMask([x1, x2], m1))
        .toThrowError(/mask.*should be an Array/);
  });
  it('computeMask error condition: incorrect number of masks', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-2, -1], [-4, -3]]);
    const addLayer = tfl.layers.add({});
    const m1 = tensor1d([true, false], 'bool');
    expect(() => addLayer.computeMask([x1, x2], [m1]))
        .toThrowError(/ are expected to have the same/);
  });
});

describeMathCPUAndGPU('Multiply Layer: Tensor', () => {
  it('2D times 2D', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const multipyLayer = tfl.layers.multiply({});
    const y = multipyLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-10, -40], [-90, -160]], [2, 2]));
  });
  // TODO(cais): Reinstate when this issue is fixed:
  //   https://github.com/PAIR-code/deeplearnjs/issues/457
  // it('2D times 2D, with broadcast', () => {
  //   const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
  //   const x2 = tensor2d([[-2], [-4]], [2, 1]);
  //   const multiplyLayer = new Multiply({});
  //   const y = multiplyLayer.apply([x1, x2]) as Tensor;
  //   expectTensorsClose(y, tensor2d([[-20, -40], [-120, -160]], [2, 2]));
  // });
});

describeMathCPUAndGPU('Average Layer: Tensor', () => {
  it('2D and 2D', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-2, -4], [-6, -8]], [2, 2]);
    const averageLayer = tfl.layers.average({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[4, 8], [12, 16]], [2, 2]));
  });
  it('2D and 2D, with broadcast', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const averageLayer = tfl.layers.average({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[4, 9], [13, 18]], [2, 2]));
  });
});

describeMathCPUAndGPU('Maximum Layer: Tensor', () => {
  it('2D and 2D', () => {
    const x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
    const x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
    const averageLayer = tfl.layers.maximum({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[10, 20], [30, 40]], [2, 2]));
  });
});

describeMathCPUAndGPU('Minimum Layer: Tensor', () => {
  it('2D and 2D', () => {
    const x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
    const x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
    const averageLayer = tfl.layers.minimum({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-2, -4], [-6, -8]], [2, 2]));
  });
});

describeMathCPUAndGPU('Concatenate Layer: Tensor', () => {
  let x1: Tensor2D;
  let x2: Tensor2D;

  function createData() {
    x1 = tensor2d([1, 2, 3, 4], [2, 2]);
    x2 = tensor2d([-1, -2, -3, -4], [2, 2]);
  }

  const axisValues: number[] = [null, undefined, 0, 1, -1];
  for (const axis of axisValues) {
    it(`axis=${axis}`, () => {
      createData();
      const layer = tfl.layers.concatenate({axis});
      const expected = axis === 0 ?
          tensor2d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2]) :
          tensor2d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4]);
      expectTensorsClose(layer.apply([x1, x2]) as Tensor, expected);
    });
  }

  it('computeMask', () => {
    const layer = tfl.layers.concatenate();
    const x1 = tensor2d([[1], [0], [1]]);
    const x2 = tensor2d([[1], [0], [0]]);
    const mask =
        layer.computeMask([x1, x2], [x1.asType('bool'), x2.asType('bool')]);
    expectTensorsClose(mask, tensor1d([true, false, false], 'bool'));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // input1 = keras.Input(shape=[4])
  // input2 = keras.Input(shape=[4])
  // y1 = keras.layers.Embedding(10,
  //                             3,
  //                             input_length=4,
  //                             mask_zero=True,
  //                             embeddings_initializer='ones')(input1)
  // y1 = keras.layers.LSTM(3,
  //                       recurrent_initializer='ones',
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y1)
  // y2 = keras.layers.Embedding(10,
  //                             3,
  //                             input_length=4,
  //                             mask_zero=True,
  //                             embeddings_initializer='ones')(input2)
  // y2 = keras.layers.LSTM(3,
  //                       recurrent_initializer='ones',
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y2)
  //
  // y = keras.layers.Concatenate()([y1, y2])
  // y = keras.layers.Dense(1,
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y)
  //
  // model = keras.Model(inputs=[input1, input2], outputs=y)
  // model.summary()
  //
  // xs1 = np.array([[0, 0, 0, 0],
  //                 [1, 0, 0, 0],
  //                 [1, 2, 0, 0],
  //                 [1, 2, 3, 0]])
  // xs2 = np.array([[0, 0, 0, 0],
  //                 [0, 0, 0, 0],
  //                 [1, 0, 0, 0],
  //                 [1, 2, 0, 0]])
  //
  // ys = model.predict([xs1, xs2])
  // print(ys)
  // ```
  it('With masking', () => {
    const input1 = tfl.input({shape: [4]});
    const input2 = tfl.input({shape: [4]});
    let y1 = tfl.layers
                 .embedding({
                   inputDim: 10,
                   outputDim: 3,
                   inputLength: 4,
                   maskZero: true,
                   embeddingsInitializer: 'ones'
                 })
                 .apply(input1) as SymbolicTensor;
    y1 = tfl.layers
             .lstm({
               units: 3,
               recurrentInitializer: 'ones',
               kernelInitializer: 'ones',
               biasInitializer: 'zeros'
             })
             .apply(y1) as SymbolicTensor;
    let y2 = tfl.layers
                 .embedding({
                   inputDim: 10,
                   outputDim: 3,
                   inputLength: 4,
                   maskZero: true,
                   embeddingsInitializer: 'ones'
                 })
                 .apply(input2) as SymbolicTensor;
    y2 = tfl.layers
             .lstm({
               units: 3,
               recurrentInitializer: 'ones',
               kernelInitializer: 'ones',
               biasInitializer: 'zeros'
             })
             .apply(y2) as SymbolicTensor;
    let y = tfl.layers.concatenate().apply([y1, y2]) as SymbolicTensor;
    y = tfl.layers
            .dense(
                {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'})
            .apply(y) as SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: y});

    const xs1 =
        tensor2d([[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0]]);
    // Notice the mask of xs2 is different from that of xs1.
    const xs2 =
        tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0]]);
    const ys = model.predict([xs1, xs2]) as Tensor;
    expectTensorsClose(
        ys, tensor2d([[0], [2.2785282], [5.169547], [5.8760333]]));
  });
});

describeMathCPU('Deserialize Merge Layers', () => {
  it('LayersModel with Add Layer', () => {
    // The following model config JSON can be obtained with Python code:
    // ```python
    // import keras
    //
    // input1 = keras.Input(shape=[4])
    // input2 = keras.Input(shape=[4])
    //
    // output = keras.layers.add([input1, input2])
    // model = keras.Model([input1, input2], output)
    //
    // model_json = model.to_json()
    // print(model_json)

    const modelWithMergeJSON: {} = {
      'class_name': 'Model',
      'keras_version': '2.1.5',
      'config': {
        'layers': [
          {
            'class_name': 'InputLayer',
            'config': {
              'dtype': 'float32',
              'batch_input_shape': [null, 4],
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
              'batch_input_shape': [null, 4],
              'name': 'input_2',
              'sparse': false
            },
            'inbound_nodes': [],
            'name': 'input_2'
          },
          {
            'class_name': 'Add',
            'config': {'trainable': true, 'name': 'add_1'},
            'inbound_nodes': [[['input_1', 0, 0, {}], ['input_2', 0, 0, {}]]],
            'name': 'add_1'
          }
        ],
        'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
        'output_layers': [['add_1', 0, 0]],
        'name': 'model_1'
      },
      'backend': 'tensorflow'
    };

    const tsConfig =
        convertPythonicToTs(modelWithMergeJSON) as serialization.ConfigDict;
    const model = deserialize(tsConfig) as tfl.LayersModel;
    expect(model.inputs.length).toEqual(2);
    expect(model.inputs[0].shape).toEqual([null, 4]);
    expect(model.inputs[1].shape).toEqual([null, 4]);
    expect(model.layers.length).toEqual(3);
    expect(model.layers[2] instanceof Add);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 4]);
  });

  it('LayersModel with Concatenate Layer', () => {
    // The following model config JSON can be obtained with Python code:
    // ```python
    // import keras
    //
    // input1 = keras.Input(shape=[4])
    // input2 = keras.Input(shape=[4])
    //
    // output = keras.layers.concatenate([input1, input2])
    // model = keras.Model([input1, input2], output)
    //
    // model_json = model.to_json()
    // print(model_json)

    const modelWithMergeJSON: {} = {
      'class_name': 'Model',
      'keras_version': '2.1.5',
      'config': {
        'layers': [
          {
            'class_name': 'InputLayer',
            'config': {
              'dtype': 'float32',
              'batch_input_shape': [null, 4],
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
              'batch_input_shape': [null, 4],
              'name': 'input_2',
              'sparse': false
            },
            'inbound_nodes': [],
            'name': 'input_2'
          },
          {
            'class_name': 'Concatenate',
            'config': {'trainable': true, 'name': 'concatenate_1', 'axis': -1},
            'inbound_nodes': [[['input_1', 0, 0, {}], ['input_2', 0, 0, {}]]],
            'name': 'concatenate_1'
          }
        ],
        'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
        'output_layers': [['concatenate_1', 0, 0]],
        'name': 'model_1'
      },
      'backend': 'tensorflow'
    };

    const tsConfig =
        convertPythonicToTs(modelWithMergeJSON) as serialization.ConfigDict;
    const model = deserialize(tsConfig) as tfl.LayersModel;
    expect(model.inputs.length).toEqual(2);
    expect(model.inputs[0].shape).toEqual([null, 4]);
    expect(model.inputs[1].shape).toEqual([null, 4]);
    expect(model.layers.length).toEqual(3);
    expect(model.layers[2] instanceof Concatenate);
    expect(model.outputs.length).toEqual(1);
    expect(model.outputs[0].shape).toEqual([null, 8]);
  });
});

describeMathCPU('Dot-Layer: Symbolic', () => {
  // Example refernce Python Keras code:
  //
  // ```py
  // import keras
  //
  // x1 = keras.Input(shape=[3, 4])
  // x2 = keras.Input(shape=[3])
  // dot_layer = keras.layers.Dot(1)
  // y = dot_layer([x1, x2])
  //
  // print(x1.shape)
  // print(x2.shape)
  // print(y.shape)
  // ```

  it('2D x 2D', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 8], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 8], null, [], null);
    const y1 = tfl.layers.dot({axes: -1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y1.shape).toEqual([null, 1]);
    const y2 = tfl.layers.dot({axes: 1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 1]);
  });

  it('3D x 3D, axes = -1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const y = tfl.layers.dot({axes: -1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y.shape).toEqual([null, 2, 2]);
  });

  it('3D x 3D, axes = 1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const y2 = tfl.layers.dot({axes: 1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 3, 3]);
  });

  it('3D x 3D, axes = 2', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const y2 = tfl.layers.dot({axes: 2}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 2, 2]);
  });

  it('2D x 3D, axes = -1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const y2 = tfl.layers.dot({axes: -1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 2]);
  });

  it('2D x 3D, axes = 1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 3, 4], null, [], null);
    const y2 = tfl.layers.dot({axes: 1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 4]);
  });

  it('3D x 2D, axes = -1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 3], null, [], null);
    const y2 = tfl.layers.dot({axes: -1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 2]);
  });

  it('3D x 2D, axes = -1', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 3, 4], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 3], null, [], null);
    const y2 = tfl.layers.dot({axes: 1}).apply([x1, x2]) as tfl.SymbolicTensor;
    expect(y2.shape).toEqual([null, 4]);
  });

  it('computeOutputShape() does not alter input shape', () => {
    const dotLayer = tfl.layers.dot({axes: 1});
    const inputShape1: Shape = [null, 3, 4];
    const inputShape2: Shape = [null, 3];
    const outputShape = dotLayer.computeOutputShape([inputShape1, inputShape2]);
    expect(outputShape).toEqual([null, 4]);
    expect(inputShape1).toEqual([null, 3, 4]);
    expect(inputShape2).toEqual([null, 3]);
  });

  // TODO(cais): Uncomment the follow test case when 4D and higher is supported
  //   by the Dot layer.
  // it('4D x 4D, axes = -1', () => {
  //   const x1 = new tfl.SymbolicTensor(
  //       'float32', [null, 2, 3, 4], null, [], null);
  //   const x2 = new tfl.SymbolicTensor(
  //       'float32', [null, 2, 3, 4], null, [], null);
  //   const y = tfl.layers.dot({axes: -1}).apply([x1, x2]) as
  //       tfl.SymbolicTensor;
  //   expect(y.shape).toEqual([null, 2, 3, 2, 3]);
  // });

  it('Dimension mismatch leads to error', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
    expect(() => tfl.layers.dot({axes: -1}).apply([
      x1, x2
    ])).toThrowError('Dimension incompatibility: 3 !== 4');
  });

  it('Incorrect number of inputs leads to error', () => {
    const x1 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x2 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    const x3 = new tfl.SymbolicTensor('float32', [null, 2, 3], null, [], null);
    expect(() => tfl.layers.dot({axes: -1}).apply([x1]))
        .toThrowError(/should be called on a list of exactly 2 inputs/);
    expect(() => tfl.layers.dot({axes: -1}).apply(x1))
        .toThrowError(/should be called on a list of exactly 2 inputs/);
    expect(() => tfl.layers.dot({axes: -1}).apply([
      x1, x2, x3
    ])).toThrowError(/should be called on a list of exactly 2 inputs/);
  });

  it('Serialization round trip', () => {
    const layer = tfl.layers.dot({axes: -1, normalize: true});
    const pythonicConfig = convertTsToPythonic(layer.getConfig());
    // tslint:disable-next-line:no-any
    const tsConfig = convertPythonicToTs(pythonicConfig) as any;
    const layerPrime = tfl.layers.dot(tsConfig);
    expect(layerPrime.getConfig().axes).toEqual(-1);
    expect(layerPrime.getConfig().normalize).toEqual(true);
  });
});

describeMathCPUAndGPU('Dot-Layer: Tensor', () => {
  // Example reference Python Keras code:
  //
  // ```py
  // import keras
  // import numpy as np
  //
  // x1 = keras.Input(shape=[2])
  // x2 = keras.Input(shape=[2])
  // dot_layer = keras.layers.Dot(-11)
  // y = dot_layer([x1, x2])
  //
  // model = keras.Model([x1, x2], y)
  // model.summary()
  //
  // xs1 = np.array([[10, 20], [30, 40]], dtype=np.float32)
  // xs2 = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
  // print(model.predict([xs1, xs2]))
  // ```

  it('2D x 2D, axis = -1', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-1, -2], [-3, -4]]);
    const dotLayer = tfl.layers.dot({axes: -1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-50], [-250]]));
  });

  it('2D x 2D, axis = -1, normalize = true', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-1, -2], [-4, -3]]);
    const dotLayer = tfl.layers.dot({axes: -1, normalize: true});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-1], [-0.96]]));
  });

  it('2D x 2D, axis = 1', () => {
    const x1 = tensor2d([[10, 20], [30, 40]]);
    const x2 = tensor2d([[-1, -2], [-3, -4]]);
    const dotLayer = tfl.layers.dot({axes: 1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-50], [-250]]));
  });

  it('3D x 2D, axis = -1', () => {
    const x1 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const x2 = tensor2d([[-1, -2], [-3, -4]]);
    const dotLayer = tfl.layers.dot({axes: -1});
    const y1 = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y1, tensor2d([[-50, -110], [-24, -10]]));
    const x3 = tensor2d([[1, 2], [3, 4]]);
    const y2 = dotLayer.apply([x1, x3]) as Tensor;
    expectTensorsClose(y2, tensor2d([[50, 110], [24, 10]]));
  });

  it('2D x 3D, axis = -1', () => {
    const x1 = tensor2d([[-1, -2], [-3, -4]]);
    const x2 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const dotLayer = tfl.layers.dot({axes: -1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-50, -110], [-24, -10]]));
  });

  it('2D x 3D, axis = 1', () => {
    const x1 = tensor2d([[-1, -2], [-3, -4]]);
    const x2 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const dotLayer = tfl.layers.dot({axes: 1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[-70, -100], [-20, -13]]));
  });

  it('3D x 3D, axis = -1', () => {
    const x1 = tensor3d([[[-1, -2], [-3, -4]], [[5, 6], [7, 8]]]);
    const x2 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const dotLayer = tfl.layers.dot({axes: -1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(
        y, tensor3d([[[-50, -110], [-110, -250]], [[38, 16], [52, 22]]]));
  });

  it('3D x 3D, axis = 1', () => {
    const x1 = tensor3d([[[-1, -2], [-3, -4]], [[5, 6], [7, 8]]]);
    const x2 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const dotLayer = tfl.layers.dot({axes: 1});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(
        y, tensor3d([[[-100, -140], [-140, -200]], [[34, 22], [40, 26]]]));
  });

  it('3D x 3D, axis = [1, 2]', () => {
    const x1 = tensor3d([[[-1, -2], [-3, -4]], [[5, 6], [7, 8]]]);
    const x2 = tensor3d([[[10, 20], [30, 40]], [[4, 3], [2, 1]]]);
    const dotLayer = tfl.layers.dot({axes: [1, 2]});
    const y = dotLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(
        y, tensor3d([[[-70, -150], [-100, -220]], [[41, 17], [48, 20]]]));
  });

  // Reference Python code:
  // ```py
  // import keras
  // import numpy as np
  //
  // input1 = keras.Input(shape=[4])
  // input2 = keras.Input(shape=[4])
  // y1 = keras.layers.Embedding(10,
  //                             3,
  //                             input_length=4,
  //                             mask_zero=True,
  //                             embeddings_initializer='ones')(input1)
  // y1 = keras.layers.LSTM(3,
  //                       recurrent_initializer='ones',
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y1)
  // y2 = keras.layers.Embedding(10,
  //                             3,
  //                             input_length=4,
  //                             mask_zero=True,
  //                             embeddings_initializer='ones')(input2)
  // y2 = keras.layers.LSTM(3,
  //                       recurrent_initializer='ones',
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y2)
  //
  // y = keras.layers.Dot(axes=[-1, -1])([y1, y2])
  // y = keras.layers.Dense(1,
  //                       kernel_initializer='ones',
  //                       bias_initializer='zeros')(y)
  //
  // model = keras.Model(inputs=[input1, input2], outputs=y)
  //
  // xs1 = np.array([[0, 0, 0, 0],
  //                 [1, 0, 0, 0],
  //                 [1, 2, 0, 0],
  //                 [1, 2, 3, 0]])
  // xs2 = np.array([[0, 0, 0, 0],
  //                 [0, 0, 0, 0],
  //                 [1, 0, 0, 0],
  //                 [1, 2, 0, 0]])
  // ys = model.predict([xs1, xs2])
  // print(ys)
  // ```
  it('With masking', () => {
    const input1 = tfl.input({shape: [4]});
    const input2 = tfl.input({shape: [4]});
    let y1 = tfl.layers
                 .embedding({
                   inputDim: 10,
                   outputDim: 3,
                   inputLength: 4,
                   maskZero: true,
                   embeddingsInitializer: 'ones'
                 })
                 .apply(input1) as SymbolicTensor;
    y1 = tfl.layers
             .lstm({
               units: 3,
               recurrentInitializer: 'ones',
               kernelInitializer: 'ones',
               biasInitializer: 'zeros'
             })
             .apply(y1) as SymbolicTensor;
    let y2 = tfl.layers
                 .embedding({
                   inputDim: 10,
                   outputDim: 3,
                   inputLength: 4,
                   maskZero: true,
                   embeddingsInitializer: 'ones'
                 })
                 .apply(input2) as SymbolicTensor;
    y2 = tfl.layers
             .lstm({
               units: 3,
               recurrentInitializer: 'ones',
               kernelInitializer: 'ones',
               biasInitializer: 'zeros'
             })
             .apply(y2) as SymbolicTensor;
    let y = tfl.layers.dot({axes: [-1, -1]}).apply([y1, y2]) as SymbolicTensor;
    y = tfl.layers
            .dense(
                {units: 1, kernelInitializer: 'ones', biasInitializer: 'zeros'})
            .apply(y) as SymbolicTensor;
    const model = tfl.model({inputs: [input1, input2], outputs: y});

    const xs1 =
        tensor2d([[0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0]]);
    // Notice the mask of xs2 is different from that of xs1.
    const xs2 =
        tensor2d([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 2, 0, 0]]);
    const ys = model.predict([xs1, xs2]) as Tensor;
    expectTensorsClose(ys, tensor2d([[0], [0], [2.195756], [2.8765779]]));
  });
});
