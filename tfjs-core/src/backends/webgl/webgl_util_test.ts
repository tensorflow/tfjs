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

import * as tf from '../../index';
import {describeWithFlags} from '../../jasmine_util';
import * as util from '../../util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import * as webgl_util from './webgl_util';

describeWithFlags('getTextureShapeFromLogicalShape', WEBGL_ENVS, () => {
  it('scalar', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([]);
    expect(texShape).toEqual([1, 1]);
  });

  it('1d', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([4]);
    expect(texShape).toEqual([1, 4]);
  });

  it('2d stays same', () => {
    let texShape = webgl_util.getTextureShapeFromLogicalShape([5, 2]);
    expect(texShape).toEqual([5, 2]);

    texShape = webgl_util.getTextureShapeFromLogicalShape([5, 1]);
    expect(texShape).toEqual([5, 1]);

    texShape = webgl_util.getTextureShapeFromLogicalShape([1, 5]);
    expect(texShape).toEqual([1, 5]);
  });

  it('3d 2x3x4', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([2, 3, 4]);
    expect(texShape).toEqual([6, 4]);
  });

  it('3d 3x256x256', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([3, 256, 256]);
    expect(texShape).toEqual([3 * 256, 256]);
  });

  it('3d 2x1x4 got squeezed', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([2, 1, 4]);
    expect(texShape).toEqual([2, 4]);
  });

  it('3d 1x8x2 got squeezed', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([1, 8, 2]);
    expect(texShape).toEqual([8, 2]);
  });

  it('4d 2x2x256x256 got squeezed', () => {
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape([2, 2, 256, 256]);
    expect(texShape).toEqual([2 * 2 * 256, 256]);
  });

  it('4d 1x8x1x3 got squeezed', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([1, 8, 1, 3]);
    expect(texShape).toEqual([8, 3]);
  });

  it('4d 1x3x1x8 got squeezed', () => {
    const texShape = webgl_util.getTextureShapeFromLogicalShape([1, 3, 1, 8]);
    expect(texShape).toEqual([3, 8]);
  });
});

describeWithFlags('getTextureShapeFromLogicalShape packed', WEBGL_ENVS, () => {
  it('textures less than 2x max size of platform preserve their shapes', () => {
    const isPacked = true;
    const logicalShape = [
      2, util.nearestLargerEven(tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE') + 1)
    ];
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(logicalShape, isPacked);
    expect(texShape).toEqual(logicalShape);
  });

  it('rows/columns do not get squeezed', () => {
    const isPacked = true;
    const logicalShape = [1, 1, 1];
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(logicalShape, isPacked);
    expect(texShape).toEqual([2, 2]);
  });

  it('squarified texture shapes account for packing constraints', () => {
    const isPacked = true;
    const max = tf.ENV.getNumber('WEBGL_MAX_TEXTURE_SIZE');

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', 5);
    const logicalShape = [1, 12];
    const texShape =
        webgl_util.getTextureShapeFromLogicalShape(logicalShape, isPacked);

    tf.ENV.set('WEBGL_MAX_TEXTURE_SIZE', max);
    expect(texShape).toEqual([6, 4]);
  });
});

describeWithFlags('isReshapeFree', WEBGL_ENVS, () => {
  it('is free when shapes have the same inner dimensions', () => {
    const before = [1, 2, 3];
    const after = [5, 2, 3];
    expect(webgl_util.isReshapeFree(before, after)).toBe(true);
  });

  it('is free when one of the shapes is a scalar', () => {
    const before: number[] = [];
    const after = [1, 2, 3];
    expect(webgl_util.isReshapeFree(before, after)).toBe(true);
  });

  it('is free when one of the dimensions equals 0', () => {
    const before = [1, 0];
    const after = [1, 2, 3];
    expect(webgl_util.isReshapeFree(before, after)).toBe(true);
  });

  it('is free when one shape is a vector and the final dimensions match',
     () => {
       const before = [9];
       const after = [1, 1, 9];
       expect(webgl_util.isReshapeFree(before, after)).toBe(true);
     });

  it('is free when one shape is a vector and the other has 1 row' +
         'in every batch and the final dimensions are even',
     () => {
       const before = [10];
       const after = [5, 1, 2];
       expect(webgl_util.isReshapeFree(before, after)).toBe(true);
     });

  it('is not free when one shape is a vector and the final dimensions' +
         'do not match and are not even',
     () => {
       const before = [18];
       const after = [2, 1, 9];
       expect(webgl_util.isReshapeFree(before, after)).toBe(false);
     });

  it('is free if the rows are divisible by two and the columns are the same',
     () => {
       const before = [1, 2, 3];
       const after = [1, 4, 3];
       expect(webgl_util.isReshapeFree(before, after)).toBe(true);
     });

  it('is not free when the inner dimensions are different and even', () => {
    const before = [1, 2, 4];
    const after = [1, 8, 10];
    expect(webgl_util.isReshapeFree(before, after)).toBe(false);
  });

  it('is not free when the inner dimensions are different and not all even',
     () => {
       const before = [1, 2, 3];
       const after = [1, 3, 2];
       expect(webgl_util.isReshapeFree(before, after)).toBe(false);
     });
});
