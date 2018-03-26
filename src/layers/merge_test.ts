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

// tslint:disable:max-line-length
import {Tensor, Tensor2D, tensor2d, tensor3d} from '@tensorflow/tfjs-core';

import {Input, Layer} from '../engine/topology';
import {DType, Shape} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Add, add, Average, average, Concatenate, concatenate, Maximum, maximum, Minimum, minimum, Multiply, multiply} from './merge';

// tslint:enable:max-line-length

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
          const addLayer = new layer({name: 'Add'});
          const symbolicInputs: SymbolicTensor[] = [];
          for (let i = 0; i < numInputs; ++i) {
            symbolicInputs.push(
                new SymbolicTensor(DType.float32, inputShape, null, [], null));
          }
          const output = addLayer.apply(symbolicInputs) as SymbolicTensor;
          expect(output.dtype).toEqual(symbolicInputs[0].dtype);
          expect(output.shape).toEqual(inputShape);
        });
      }
    }
  }

  it('Single input leads to exception', () => {
    const x = new SymbolicTensor(DType.float32, [2, 2], null, [], null);
    const addLayer = new Add({name: 'Add'});
    expect(() => {
      addLayer.apply([x]);
    }).toThrowError(/.*at least 2 inputs\. Got 1 input.*/);
  });

  it('Non-unique batch sizes to exception', () => {
    const x1 = new SymbolicTensor(DType.float32, [1, 2], null, [], null);
    const x2 = new SymbolicTensor(DType.float32, [2, 2], null, [], null);
    const addLayer = new Add({name: 'Add'});
    expect(() => {
      addLayer.apply([x1, x2]);
    }).toThrowError(/Can not merge tensors with different batch sizes/);
  });
});

describeMathCPUAndGPU('Add-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(add().constructor.name).toEqual('Add');
  });

  it('Calling with config arg returns Layer', () => {
    expect((add({name: 'addLayer'}) as Layer).name.indexOf('addLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 2]});
    const input2 = Input({shape: [2, 2]});
    const output = add([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = add([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([11, 22, 33, 44], [2, 2]));
  });
});

describeMathCPUAndGPU('Multiply-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(multiply().constructor.name).toEqual('Multiply');
  });

  it('Calling with config arg returns Layer', () => {
    expect((multiply({name: 'multiplyLayer'}) as Layer)
               .name.indexOf('multiplyLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 2]});
    const input2 = Input({shape: [2, 2]});
    const output = multiply([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = multiply([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([10, 40, 90, 160], [2, 2]));
  });
});

describeMathCPUAndGPU('Average-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(average().constructor.name).toEqual('Average');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        (average({name: 'averageLayer'}) as Layer).name.indexOf('averageLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 2]});
    const input2 = Input({shape: [2, 2]});
    const output = average([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 2, 3, 4], [2, 2]);
    const input2 = tensor2d([10, 20, 30, 40], [2, 2]);
    const output = average([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([5.5, 11, 16.5, 22], [2, 2]));
  });
});

describeMathCPUAndGPU('Maximum-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(maximum().constructor.name).toEqual('Maximum');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        (maximum({name: 'maximumLayer'}) as Layer).name.indexOf('maximumLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 2]});
    const input2 = Input({shape: [2, 2]});
    const output = maximum([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 20, 3, 40], [2, 2]);
    const input2 = tensor2d([10, 2, 30, 4], [2, 2]);
    const output = maximum([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([10, 20, 30, 40], [2, 2]));
  });
});

describeMathCPUAndGPU('Minimum-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(minimum().constructor.name).toEqual('Minimum');
  });

  it('Calling with config arg returns Layer', () => {
    expect(
        (minimum({name: 'minimumLayer'}) as Layer).name.indexOf('minimumLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 2]});
    const input2 = Input({shape: [2, 2]});
    const output = minimum([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 2]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([1, 20, 3, 40], [2, 2]);
    const input2 = tensor2d([10, 2, 30, 4], [2, 2]);
    const output = minimum([input1, input2]) as Tensor;
    expectTensorsClose(output, tensor2d([1, 2, 3, 4], [2, 2]));
  });
});

describeMathCPUAndGPU('Concatenate-Functional', () => {
  it('Calling without arg returns Layer', () => {
    expect(concatenate().constructor.name).toEqual('Concatenate');
  });

  it('Calling with config arg returns Layer', () => {
    expect((concatenate({name: 'concatenateLayer'}) as Layer)
               .name.indexOf('concatenateLayer'))
        .toEqual(0);
  });

  it('Calling with symbolic tensors returns symbolic tensor', () => {
    const input1 = Input({shape: [2, 3]});
    const input2 = Input({shape: [2, 4]});
    const output = concatenate([input1, input2]) as SymbolicTensor;
    expect(output.shape).toEqual([null, 2, 7]);
  });

  it('Calling with tensors returns tensor', () => {
    const input1 = tensor2d([[1, 2], [3, 4]], [2, 2]);
    const input2 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const output = concatenate([input1, input2]) as Tensor;
    expectTensorsClose(
        output, tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]));
  });
});


describeMathCPU('Concatenate Layer: Symbolic', () => {
  it('All known shapes', () => {
    const x1 = new SymbolicTensor(DType.float32, [2, 3, 4], null, [], null);
    const x2 = new SymbolicTensor(DType.float32, [2, 3, 4], null, [], null);
    const layer0 = new Concatenate({});
    expect((layer0.apply([x1, x2]) as SymbolicTensor).shape).toEqual([2, 3, 8]);
    const layer1 = new Concatenate({axis: -1});
    expect((layer1.apply([x1, x2]) as SymbolicTensor).shape).toEqual([2, 3, 8]);
    const layer2 = new Concatenate({axis: 0});
    expect((layer2.apply([x1, x2]) as SymbolicTensor).shape).toEqual([4, 3, 4]);
    const layer3 = new Concatenate({axis: 1});
    expect((layer3.apply([x1, x2]) as SymbolicTensor).shape).toEqual([2, 6, 4]);
  });
  it('Concat axis has unknown shape', () => {
    const x1 = new SymbolicTensor(DType.float32, [2, null, 4], null, [], null);
    const x2 = new SymbolicTensor(DType.float32, [2, null, 4], null, [], null);
    const layer = new Concatenate({axis: 1});
    expect((layer.apply([x1, x2]) as SymbolicTensor).shape).toEqual([
      2, null, 4
    ]);
  });
  it('Non-concat axis has unknown shape', () => {
    const x1 = new SymbolicTensor(DType.float32, [null, 3, 4], null, [], null);
    const x2 = new SymbolicTensor(DType.float32, [null, 5, 4], null, [], null);
    const layer = new Concatenate({axis: 1});
    expect((layer.apply([x1, x2]) as SymbolicTensor).shape).toEqual([
      null, 8, 4
    ]);
  });
  it('Incompatible shape leads to error', () => {
    const x1 = new SymbolicTensor(DType.float32, [2, 3, 5], null, [], null);
    const x2 = new SymbolicTensor(DType.float32, [2, 4, 5], null, [], null);
    const layer = new Concatenate({});
    expect(() => layer.apply([
      x1, x2
    ])).toThrowError(/requires inputs with matching shapes except/);
  });
  it('Single shape leads to error', () => {
    const x1 = new SymbolicTensor(DType.float32, [2, 3, 5], null, [], null);
    const layer = new Concatenate({});
    expect(() => layer.apply([x1]))
        .toThrowError(/should be called on a list of at least 2 inputs/);
  });
});

describeMathCPUAndGPU('Add Layer: Tensor', () => {
  it('2D plus 2D', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const addLayer = new Add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[9, 18], [27, 36]], [2, 2]));
  });
  it('2D plus 2D, with broadcast', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const addLayer = new Add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[8, 18], [26, 36]], [2, 2]));
  });
  it('2D plus 2D, with dimension expansion', () => {
    const x1 =
        tensor3d([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], [2, 2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const addLayer = new Add({});
    const y = addLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(
        y, tensor3d([[[8, 18], [28, 38]], [[46, 56], [66, 76]]], [2, 2, 2]));
  });
});

describeMathCPUAndGPU('Multiply Layer: Tensor', () => {
  it('2D times 2D', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
    const multipyLayer = new Multiply({});
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
    const averageLayer = new Average({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[4, 8], [12, 16]], [2, 2]));
  });
  it('2D and 2D, with broadcast', () => {
    const x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
    const x2 = tensor2d([[-2], [-4]], [2, 1]);
    const averageLayer = new Average({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[4, 9], [13, 18]], [2, 2]));
  });
});

describeMathCPUAndGPU('Maximum Layer: Tensor', () => {
  it('2D and 2D', () => {
    const x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
    const x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
    const averageLayer = new Maximum({});
    const y = averageLayer.apply([x1, x2]) as Tensor;
    expectTensorsClose(y, tensor2d([[10, 20], [30, 40]], [2, 2]));
  });
});

describeMathCPUAndGPU('Minimum Layer: Tensor', () => {
  it('2D and 2D', () => {
    const x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
    const x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
    const averageLayer = new Minimum({});
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
      const layer = new Concatenate({axis});
      const expected = axis === 0 ?
          tensor2d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2]) :
          tensor2d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4]);
      expectTensorsClose(layer.apply([x1, x2]) as Tensor, expected);
    });
  }
});
