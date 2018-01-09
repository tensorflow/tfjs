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

import * as concat_util from './concat_util';

describe('concat_util.assertConcatShapesMatch rank=3D', () => {
  it('Non-3D tensor x1', () => {
    const assertFn = () => {
      concat_util.assertParams([1], [1, 2, 3], 1);
    };

    expect(assertFn).toThrow();
  });

  it('Non-3D tensor x2', () => {
    const assertFn = () => {
      concat_util.assertParams([1, 2, 3], [2, 3], 1);
    };

    expect(assertFn).toThrow();
  });

  it('axis out of bound', () => {
    const assertFn = () => {
      concat_util.assertParams([1, 2, 3], [1, 2, 3], 4);
    };

    expect(assertFn).toThrow();
  });

  it('non-axis shape mismatch', () => {
    const assertFn = () => {
      concat_util.assertParams([2, 3, 3], [2, 2, 4], 2);
    };

    expect(assertFn).toThrow();
  });

  it('shapes line up', () => {
    const assertFn = () => {
      concat_util.assertParams([2, 3, 3], [2, 3, 4], 2);
    };

    expect(assertFn).not.toThrow();
  });
});

describe('concat_util.computeConcatOutputShape', () => {
  it('compute output shape, axis=0', () => {
    expect(concat_util.computeOutShape([2, 2, 3], [1, 2, 3], 0)).toEqual([
      3, 2, 3
    ]);
  });
});

describe('concat_util.computeBackpropSizes', () => {
  it('compute backprop sizes of 2D tensors, axis=0', () => {
    const x1 = [2, 3];
    const y = [5, 3];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes2D(x1, y, 0);
    expect(x1Begin).toEqual([0, 0]);
    expect(x1Size).toEqual([2, 3]);
    expect(x2Begin).toEqual([2, 0]);
    expect(x2Size).toEqual([3, 3]);
  });

  it('compute backprop sizes of 2D tensors, axis=1', () => {
    const x1 = [2, 3];
    const y = [2, 7];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes2D(x1, y, 1);
    expect(x1Begin).toEqual([0, 0]);
    expect(x1Size).toEqual([2, 3]);
    expect(x2Begin).toEqual([0, 3]);
    expect(x2Size).toEqual([2, 4]);
  });

  it('compute backprop sizes of 3D tensors, axis=0', () => {
    const x1 = [2, 3, 2];
    const y = [5, 3, 2];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes3D(x1, y, 0);
    expect(x1Begin).toEqual([0, 0, 0]);
    expect(x1Size).toEqual([2, 3, 2]);
    expect(x2Begin).toEqual([2, 0, 0]);
    expect(x2Size).toEqual([3, 3, 2]);
  });

  it('compute backprop sizes of 3D tensors, axis=1', () => {
    const x1 = [2, 3, 2];
    const y = [2, 7, 2];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes3D(x1, y, 1);
    expect(x1Begin).toEqual([0, 0, 0]);
    expect(x1Size).toEqual([2, 3, 2]);
    expect(x2Begin).toEqual([0, 3, 0]);
    expect(x2Size).toEqual([2, 4, 2]);
  });

  it('compute backprop sizes of 3D tensors, axis=2', () => {
    const x1 = [2, 3, 2];
    const y = [2, 3, 3];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes3D(x1, y, 2);
    expect(x1Begin).toEqual([0, 0, 0]);
    expect(x1Size).toEqual([2, 3, 2]);
    expect(x2Begin).toEqual([0, 0, 2]);
    expect(x2Size).toEqual([2, 3, 1]);
  });

  it('compute backprop sizes of 4D tensors, axis=0', () => {
    const x1 = [2, 3, 2, 4];
    const y = [3, 3, 2, 4];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes4D(x1, y, 0);
    expect(x1Begin).toEqual([0, 0, 0, 0]);
    expect(x1Size).toEqual([2, 3, 2, 4]);
    expect(x2Begin).toEqual([2, 0, 0, 0]);
    expect(x2Size).toEqual([1, 3, 2, 4]);
  });

  it('compute backprop sizes of 4D tensors, axis=1', () => {
    const x1 = [3, 3, 2, 4];
    const y = [3, 4, 2, 4];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes4D(x1, y, 1);
    expect(x1Begin).toEqual([0, 0, 0, 0]);
    expect(x1Size).toEqual([3, 3, 2, 4]);
    expect(x2Begin).toEqual([0, 3, 0, 0]);
    expect(x2Size).toEqual([3, 1, 2, 4]);
  });

  it('compute backprop sizes of 4D tensors, axis=2', () => {
    const x1 = [3, 3, 2, 4];
    const y = [3, 3, 4, 4];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes4D(x1, y, 2);
    expect(x1Begin).toEqual([0, 0, 0, 0]);
    expect(x1Size).toEqual([3, 3, 2, 4]);
    expect(x2Begin).toEqual([0, 0, 2, 0]);
    expect(x2Size).toEqual([3, 3, 2, 4]);
  });

  it('compute backprop sizes of 4D tensors, axis=3', () => {
    const x1 = [2, 3, 2, 4];
    const y = [2, 3, 2, 8];
    const {x1Begin, x1Size, x2Begin, x2Size} =
        concat_util.computeGradientSliceShapes4D(x1, y, 3);
    expect(x1Begin).toEqual([0, 0, 0, 0]);
    expect(x1Size).toEqual([2, 3, 2, 4]);
    expect(x2Begin).toEqual([0, 0, 0, 4]);
    expect(x2Size).toEqual([2, 3, 2, 4]);
  });
});
