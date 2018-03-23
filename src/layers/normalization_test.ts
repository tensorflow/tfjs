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
 * Unit tests for normalization layers.
 */

// tslint:disable:max-line-length
import {Tensor, tensor2d, tensor3d, tensor4d} from '@tensorflow/tfjs-core';

import {DType} from '../types';
import {SymbolicTensor} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {BatchNormalization} from './normalization';
// tslint:enable

describeMathCPU('BatchNormalization Layers: Symbolic', () => {
  const validInputShapes = [[4, 6], [2, 3, 4], [2, 3, 4, 5]];
  for (const inputShape of validInputShapes) {
    const testTitle = `shape=${JSON.stringify(inputShape)}`;
    it(testTitle, () => {
      const x = new SymbolicTensor(DType.float32, inputShape, null, [], null);
      const layer = new BatchNormalization({});
      const y = layer.apply(x) as SymbolicTensor;
      expect(y.dtype).toEqual(x.dtype);
      expect(y.shape).toEqual(x.shape);
    });
  }

  it('Undetermined dim axis leads to ValueError', () => {
    const x = new SymbolicTensor(DType.float32, [null, 2, 3], null, [], null);
    const layer = new BatchNormalization({axis: 0});
    expect(() => layer.apply(x))
        .toThrowError(
            /Axis 0 of input tensor should have a defined dimension.*/);
  });
});

describeMathCPUAndGPU('BatchNormalization Layers: Tensor', () => {
  const dimensions = [2, 3, 4];
  const axisValues = [0, -1];

  for (const dim of dimensions) {
    for (const axis of axisValues) {
      const testTitle = `Inference, ${dim}D, axis=${axis}`;
      it(testTitle, () => {
        const layer = new BatchNormalization({axis});
        let x: Tensor;
        if (dim === 2) {
          x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        } else if (dim === 3) {
          x = tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]], [2, 2, 2]);
        } else if (dim === 4) {
          x = tensor4d(
              [
                [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]],
                [[[-1, -2], [-3, -4]], [[1, 2], [3, 4]]]
              ],
              [2, 2, 2, 2]);
        }
        const y = layer.apply(x, {training: false}) as Tensor;
        expectTensorsClose(y, x, 0.01);
      });
    }
  }

  // TODO(cais): Test BatchNormalization under training model.
});
