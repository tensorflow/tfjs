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

import {describeWithFlags} from '../../jasmine_util';
import {expectArraysClose} from '../../test_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import * as tex_util from './tex_util';

describe('tex_util getUnpackedMatrixTextureShapeWidthHeight', () => {
  it('[1x1] => [1x1]', () => {
    expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(1, 1)).toEqual([
      1, 1
    ]);
  });

  it('[MxN] => [NxM]', () => {
    expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(123, 456))
        .toEqual([456, 123]);
  });
});

describe('tex_util getPackedMatrixTextureShapeWidthHeight', () => {
  it('[1x1] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 1);
    expect(shape).toEqual([1, 1]);
  });

  it('[1x2] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 2);
    expect(shape).toEqual([1, 1]);
  });

  it('[2x1] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 1);
    expect(shape).toEqual([1, 1]);
  });

  it('[2x2] => [1x1]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 2);
    expect(shape).toEqual([1, 1]);
  });

  it('[3x3] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 3);
    expect(shape).toEqual([2, 2]);
  });

  it('[4x3] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 3);
    expect(shape).toEqual([2, 2]);
  });

  it('[3x4] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 4);
    expect(shape).toEqual([2, 2]);
  });

  it('[4x4] => [2x2]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 4);
    expect(shape).toEqual([2, 2]);
  });

  it('[1024x1024] => [512x512]', () => {
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1024, 1024);
    expect(shape).toEqual([512, 512]);
  });

  it('[MxN] => [ceil(N/2)xceil(M/2)]', () => {
    const M = 123;
    const N = 5013;
    const shape = tex_util.getPackedMatrixTextureShapeWidthHeight(M, N);
    expect(shape).toEqual([Math.ceil(N / 2), Math.ceil(M / 2)]);
  });
});

describeWithFlags('tex_util getDenseTexShape', WEBGL_ENVS, () => {
  it('basic', () => {
    const shape = [1, 3, 3, 4];
    const denseShape = tex_util.getDenseTexShape(shape);
    expectArraysClose(denseShape, [3, 3]);
  });
});