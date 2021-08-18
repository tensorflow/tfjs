/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as conv_util from './conv_util';

describe('conv_util computeConv2DInfo', () => {
  it('1x1 conv over 1x1 array with same pad', () => {
    const inShape: [number, number, number, number] = [1, 1, 1, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [1, 1, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('2x2 conv over 3x3 array with same pad', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
    // Should produce non-even padding with extra pixel at the right/bottom.
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2 conv over 3x3 array with same pad', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x2 conv over 3x3 array with valid pad', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('3x3 conv over 5x5 array with same pad with stride 2', () => {
    const inShape: [number, number, number, number] = [1, 5, 5, 1];
    const stride = 2;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [3, 3, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(3);
    expect(convInfo.effectiveFilterHeight).toEqual(3);

    expect(convInfo.padInfo.left).toBe(1);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(1);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2 conv over 3x3 array with valid pad with stride 2', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 2;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x1 conv over 3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 1, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x1 conv over 3x3 array with valid pad with strides h=2, w=1', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const strides: [number, number] = [2, 1];
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 1, 1, 1], strides, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('1x2 conv over 3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [1, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('1x2 conv over 3x3 array with valid pad with stride 1, batch=5', () => {
    const inShape: [number, number, number, number] = [5, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [1, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(5);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('2x2 conv over 3x3 array with same pad with dilations 2', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilations, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    // pad evenly on all sides
    expect(convInfo.padInfo.left).toBe(1);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(1);
    expect(convInfo.padInfo.bottom).toBe(1);
    expect(convInfo.effectiveFilterWidth).toEqual(3);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });

  it('2x1 conv over 3x3 array with same pad with dilations 2', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 1, 1, 1], stride, dilations, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    // pad top and bottom
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(0);
    expect(convInfo.padInfo.top).toBe(1);
    expect(convInfo.padInfo.bottom).toBe(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });

  it('3x4 conv over 8x8 array with same pad with dilations h=4 w=3', () => {
    const inShape: [number, number, number, number] = [1, 8, 8, 1];
    const stride = 1;
    const dilations: [number, number] = [4, 3];
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [3, 4, 1, 1], stride, dilations, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(8);
    expect(convInfo.outWidth).toEqual(8);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(10);
    expect(convInfo.effectiveFilterHeight).toEqual(9);

    expect(convInfo.padInfo.left).toBe(4);
    expect(convInfo.padInfo.right).toBe(5);
    expect(convInfo.padInfo.top).toBe(4);
    expect(convInfo.padInfo.bottom).toBe(4);
  });

  it('2x1 conv over 3x3 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 1, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });

  it('2x2 conv over 3x3 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number] = [1, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(3);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });

  it('2x2 conv over 4x4 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number] = [1, 4, 4, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(3);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });
});

describe('conv_util computeConv3DInfo', () => {
  it('1x1x1 conv over 1x1x1 array with same pad', () => {
    const inShape: [number, number, number, number, number] = [1, 1, 1, 1, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [1, 1, 1, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x2x2 conv over 3x3x3 array with same pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    // Should produce non-even padding with extra pixel at the back/right/bottom
    expect(convInfo.padInfo.front).toBe(0);
    expect(convInfo.padInfo.back).toBe(1);
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2x2 conv over 3x3x3 array with same pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x2x2 conv over 3x3x3 array with valid pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('3x3x3 conv over 5x5x5 array with same pad with stride 2', () => {
    const inShape: [number, number, number, number, number] = [1, 5, 5, 5, 1];
    const stride = 2;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [3, 3, 3, 1, 1], stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);

    expect(convInfo.padInfo.front).toBe(1);
    expect(convInfo.padInfo.back).toBe(1);
    expect(convInfo.padInfo.left).toBe(1);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(1);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2x2 conv over 3x3x3 array with valid pad with stride 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 2;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x1x1 conv over 3x3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 1, 1, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x1x1 conv over 3x3x3 array with valid pad with strides d=2, h=1, w=1',
     () => {
       const inShape: [number, number, number, number, number] =
           [1, 3, 3, 3, 1];
       const strides: [number, number, number] = [2, 1, 1];
       const dilation = 1;
       const convInfo = conv_util.computeConv3DInfo(
           inShape, [2, 1, 1, 1, 1], strides, dilation, 'valid');
       expect(convInfo.batchSize).toEqual(1);
       expect(convInfo.outDepth).toEqual(1);
       expect(convInfo.outHeight).toEqual(3);
       expect(convInfo.outWidth).toEqual(3);
       expect(convInfo.outChannels).toEqual(1);
     });

  it('1x2x2 conv over 3x3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [1, 2, 2, 1, 1], stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('1x2x2 conv over 3x3x3 array with valid pad with stride 1, batch=5',
     () => {
       const inShape: [number, number, number, number, number] =
           [5, 3, 3, 3, 1];
       const stride = 1;
       const dilation = 1;
       const convInfo = conv_util.computeConv3DInfo(
           inShape, [1, 2, 2, 1, 1], stride, dilation, 'valid');
       expect(convInfo.batchSize).toEqual(5);
       expect(convInfo.outDepth).toEqual(3);
       expect(convInfo.outHeight).toEqual(2);
       expect(convInfo.outWidth).toEqual(2);
       expect(convInfo.outChannels).toEqual(1);
     });

  it('2x2x2 conv over 3x3x3 array with same pad with dilations 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilations, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    // pad evenly on all sides
    expect(convInfo.padInfo.front).toBe(1);
    expect(convInfo.padInfo.back).toBe(1);
    expect(convInfo.padInfo.left).toBe(1);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(1);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x1x1 conv over 3x3x3 array with same pad with dilations 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 1, 1, 1, 1], stride, dilations, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    // pad top and bottom
    expect(convInfo.padInfo.front).toBe(1);
    expect(convInfo.padInfo.back).toBe(1);
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(0);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(0);
  });

  it('3x4x4 conv over 8x8 array with same pad with dilations d=4 h=3 w=3',
     () => {
       const inShape: [number, number, number, number, number] =
           [1, 8, 8, 8, 1];
       const stride = 1;
       const dilations: [number, number, number] = [4, 3, 3];
       const convInfo = conv_util.computeConv3DInfo(
           inShape, [3, 4, 4, 1, 1], stride, dilations, 'same');
       expect(convInfo.batchSize).toEqual(1);
       expect(convInfo.outDepth).toEqual(8);
       expect(convInfo.outHeight).toEqual(8);
       expect(convInfo.outWidth).toEqual(8);
       expect(convInfo.outChannels).toEqual(1);

       expect(convInfo.padInfo.front).toBe(4);
       expect(convInfo.padInfo.back).toBe(4);
       expect(convInfo.padInfo.left).toBe(4);
       expect(convInfo.padInfo.right).toBe(5);
       expect(convInfo.padInfo.top).toBe(4);
       expect(convInfo.padInfo.bottom).toBe(5);
     });

  it('2x1x1 conv over 3x3x3 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 1, 1, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x2x2 conv over 3x3x3 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x2x2 conv over 4x4x4 array with valid pad with dilations 2', () => {
    const inShape: [number, number, number, number, number] = [1, 4, 4, 4, 1];
    const stride = 1;
    const dilations = 2;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, 1, 1], stride, dilations, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
  });
});

describe('conv_util computeConv2DInfo with depthwise=true', () => {
  it('1x1 filter over 1x1 array with same pad', () => {
    const inChannels = 1;
    const inShape: [number, number, number, number] = [1, 1, 1, inChannels];
    const fSize = 1;
    const chMul = 1;
    const stride = 1;
    const dilation = 1;
    const pad = 'same';
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad, null,
        true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('2x2 filter over 3x3 array with same pad, chMul=3, depth=2', () => {
    const inChannels = 2;
    const batchSize = 1;
    const inSize = 3;
    const inShape: [number, number, number, number] =
        [batchSize, inSize, inSize, inChannels];
    const fSize = 2;
    const chMul = 3;
    const stride = 1;
    const dilation = 1;
    const pad = 'same';
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad, null,
        true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(6);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x2 filter over 3x3 array with valid pad, chMul=3, depth=2', () => {
    const inChannels = 2;
    const batchSize = 1;
    const inSize = 3;
    const inShape: [number, number, number, number] =
        [batchSize, inSize, inSize, inChannels];
    const fSize = 2;
    const chMul = 3;
    const stride = 1;
    const dilation = 1;
    const pad = 'valid';
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad, null,
        true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(6);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });
});

describe('conv_util computeConv3DInfo with depthwise=true', () => {
  it('1x1x1 filter over 1x1x1 array with same pad', () => {
    const inChannels = 1;
    const inShape: [number, number, number, number, number] =
        [1, 1, 1, 1, inChannels];
    const fSize = 1;
    const chMul = 1;
    const stride = 1;
    const dilation = 1;
    const pad = 'same';
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [fSize, fSize, fSize, inChannels, chMul], stride, dilation,
        pad, true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
  });

  it('2x2x2 filter over 3x3x3 array with same pad, chMul=3, depth=2', () => {
    const inChannels = 2;
    const batchSize = 1;
    const inSize = 3;
    const inShape: [number, number, number, number, number] =
        [batchSize, inSize, inSize, inSize, inChannels];
    const fSize = 2;
    const chMul = 3;
    const stride = 1;
    const dilation = 1;
    const pad = 'same';
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [fSize, fSize, fSize, inChannels, chMul], stride, dilation,
        pad, true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(6);
  });

  it('2x2x2 filter over 3x3x3 array with valid pad, chMul=3, depth=2', () => {
    const inChannels = 2;
    const batchSize = 1;
    const inSize = 3;
    const inShape: [number, number, number, number, number] =
        [batchSize, inSize, inSize, inSize, inChannels];
    const fSize = 2;
    const chMul = 3;
    const stride = 1;
    const dilation = 1;
    const pad = 'valid';
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [fSize, fSize, fSize, inChannels, chMul], stride, dilation,
        pad, true);
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(6);
  });
});

describe('conv_util computeConv2DInfo channelsFirst', () => {
  it('2x2 conv over 3x3 array with same pad', () => {
    const inDepth = 2;
    const outDepth = 4;
    const inShape: [number, number, number, number] = [1, inDepth, 3, 3];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, inDepth, outDepth], stride, dilation, 'same', null,
        false, 'channelsFirst');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(4);
    expect(convInfo.outShape).toEqual([1, 4, 3, 3]);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
    // Should produce non-even padding with extra pixel at the right/bottom.
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2 conv over 3x3 array with valid pad', () => {
    const inDepth = 6;
    const outDepth = 16;
    const inShape: [number, number, number, number] = [1, inDepth, 3, 3];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [2, 2, inDepth, outDepth], stride, dilation, 'valid', null,
        false, 'channelsFirst');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(16);
    expect(convInfo.outShape).toEqual([1, 16, 2, 2]);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
    // Should produce no padding.
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(0);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(0);
  });
});

describe('conv_util computeConv3DInfo channelsFirst', () => {
  it('2x2x2 conv over 3x3x3 array with same pad', () => {
    const inDepth = 2;
    const outDepth = 4;
    const inShape: [number, number, number, number, number] =
        [1, inDepth, 3, 3, 3];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, inDepth, outDepth], stride, dilation, 'same', false,
        'channelsFirst');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(4);
    expect(convInfo.outShape).toEqual([1, 4, 3, 3, 3]);
    // Should produce non-even padding with extra pixel at the back/right/bottom
    expect(convInfo.padInfo.front).toBe(0);
    expect(convInfo.padInfo.back).toBe(1);
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2x2 conv over 3x3x3 array with valid pad', () => {
    const inDepth = 6;
    const outDepth = 16;
    const inShape: [number, number, number, number, number] =
        [1, inDepth, 3, 3, 3];
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computeConv3DInfo(
        inShape, [2, 2, 2, inDepth, outDepth], stride, dilation, 'valid', false,
        'channelsFirst');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(16);
    expect(convInfo.outShape).toEqual([1, 16, 2, 2, 2]);
    // Should produce no padding.
    expect(convInfo.padInfo.front).toBe(0);
    expect(convInfo.padInfo.back).toBe(0);
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(0);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(0);
  });
});

describe('conv_util computeConv2DInfo roundingMode', () => {
  const inChannels = 6;
  const batchSize = 1;
  const inSize = 5;
  const inShape: [number, number, number, number] =
      [batchSize, inSize, inSize, inChannels];
  const fSize = 2;
  const chMul = 12;
  const stride = 2;
  const dilation = 1;
  const pad = 1;

  it('Default truncate the output dimension of Conv Layer', () => {
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad);

    expect(convInfo.outShape).toEqual([batchSize, 3, 3, chMul]);
  });

  it('Floor the output dimension of Conv Layer', () => {
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad,
        'floor');

    expect(convInfo.outShape).toEqual([batchSize, 3, 3, chMul]);
  });

  it('Round the output dimension of Conv Layer', () => {
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad,
        'round');

    expect(convInfo.outShape).toEqual([batchSize, 4, 4, chMul]);
  });

  it('Ceil the output dimension of Conv Layer', () => {
    const convInfo = conv_util.computeConv2DInfo(
        inShape, [fSize, fSize, inChannels, chMul], stride, dilation, pad,
        'ceil');

    expect(convInfo.outShape).toEqual([batchSize, 4, 4, chMul]);
  });
});

describe('conv_util computePoolInfo roundingMode', () => {
  const inChannels = 6;
  const batchSize = 1;
  const inSize = 5;
  const inShape: [number, number, number, number] =
      [batchSize, inSize, inSize, inChannels];
  const fSize = 2;
  const stride = 2;
  const dilation = 1;
  const pad = 1;

  it('Default truncate the output dimension of Pool Layer', () => {
    const poolInfo = conv_util.computePool2DInfo(
        inShape, [fSize, fSize], stride, pad, dilation, 'floor');

    expect(poolInfo.outShape).toEqual([batchSize, 3, 3, inChannels]);
  });

  it('Floor the output dimension of Pool Layer', () => {
    const poolInfo = conv_util.computePool2DInfo(
        inShape, [fSize, fSize], stride, pad, dilation, 'floor');

    expect(poolInfo.outShape).toEqual([batchSize, 3, 3, inChannels]);
  });

  it('Round the output dimension of Pool Layer', () => {
    const poolInfo = conv_util.computePool2DInfo(
        inShape, [fSize, fSize], stride, pad, dilation, 'round');

    expect(poolInfo.outShape).toEqual([batchSize, 4, 4, inChannels]);
  });

  it('Ceil the output dimension of Pool Layer', () => {
    const poolInfo = conv_util.computePool2DInfo(
        inShape, [fSize, fSize], stride, pad, dilation, 'ceil');

    expect(poolInfo.outShape).toEqual([batchSize, 4, 4, inChannels]);
  });
});

describe('conv_util computePool3dInfo', () => {
  it('1x1x1 pool over 1x1x1 array with valid pad', () => {
    const inShape: [number, number, number, number, number] = [1, 1, 1, 1, 1];
    const filterSize = 1;
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('1x1x1 pool over 3x3x3 array with valid pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 1;
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(1);
    expect(convInfo.effectiveFilterWidth).toEqual(1);
    expect(convInfo.effectiveFilterHeight).toEqual(1);
  });

  it('2x2x2 pool over 3x3x3 array with same pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 2;
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'same');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(3);
    expect(convInfo.outHeight).toEqual(3);
    expect(convInfo.outWidth).toEqual(3);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(2);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
    expect(convInfo.padInfo.top).toEqual(0);
    expect(convInfo.padInfo.bottom).toEqual(1);
    expect(convInfo.padInfo.left).toEqual(0);
    expect(convInfo.padInfo.right).toEqual(1);
    expect(convInfo.padInfo.front).toEqual(0);
    expect(convInfo.padInfo.back).toEqual(1);
    expect(convInfo.padInfo.type).toEqual('SAME');
  });

  it('2x2x2 pool over 3x3x3 array with valid pad', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 2;
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(2);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x2x2 pool over 4x4x4 array with valid pad, stride 2', () => {
    const inShape: [number, number, number, number, number] = [1, 4, 4, 4, 1];
    const filterSize = 2;
    const stride = 2;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(2);
    expect(convInfo.outHeight).toEqual(2);
    expect(convInfo.outWidth).toEqual(2);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(2);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
  });

  it('2x2x2 pool over 3x3x3 array with valid pad, dilation 2', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 2;
    const stride = 1;
    const dilation = 2;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 'valid');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(1);
    expect(convInfo.outHeight).toEqual(1);
    expect(convInfo.outWidth).toEqual(1);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(3);
    expect(convInfo.effectiveFilterWidth).toEqual(3);
    expect(convInfo.effectiveFilterHeight).toEqual(3);
  });

  it('2x2x2 pool over 3x3x3 array with pad 1, roundingMode floor', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 2;
    const stride = 1;
    const dilation = 1;
    const convInfo = conv_util.computePool3DInfo(
        inShape, filterSize, stride, dilation, 1, 'floor');
    expect(convInfo.batchSize).toEqual(1);
    expect(convInfo.outDepth).toEqual(4);
    expect(convInfo.outHeight).toEqual(4);
    expect(convInfo.outWidth).toEqual(4);
    expect(convInfo.outChannels).toEqual(1);
    expect(convInfo.effectiveFilterDepth).toEqual(2);
    expect(convInfo.effectiveFilterWidth).toEqual(2);
    expect(convInfo.effectiveFilterHeight).toEqual(2);
    expect(convInfo.padInfo.top).toEqual(1);
    expect(convInfo.padInfo.bottom).toEqual(1);
    expect(convInfo.padInfo.left).toEqual(1);
    expect(convInfo.padInfo.right).toEqual(1);
    expect(convInfo.padInfo.front).toEqual(1);
    expect(convInfo.padInfo.back).toEqual(1);
    expect(convInfo.padInfo.type).toEqual('NUMBER');
  });

  it('throws unknown dataFormat', () => {
    const inShape: [number, number, number, number, number] = [1, 3, 3, 3, 1];
    const filterSize = 2;
    const stride = 1;
    const dilation = 1;
    const fakeDataFormat = 'fakeFormat' as 'NDHWC' | 'NCDHW';
    expect(
        () => conv_util.computePool3DInfo(
            inShape, filterSize, stride, dilation, 1, 'floor', fakeDataFormat))
        .toThrowError();
  });
});

describe('conv_util convertConv2DDataFormat', () => {
  it('convert NHWC to channelsLast', () => {
    const dataFormat: 'NHWC'|'NCHW' = 'NHWC';
    const $dataFormat = conv_util.convertConv2DDataFormat(dataFormat);
    expect($dataFormat).toEqual('channelsLast');
  });

  it('convert NCHW to channelsFirst', () => {
    const dataFormat: 'NHWC'|'NCHW' = 'NCHW';
    const $dataFormat = conv_util.convertConv2DDataFormat(dataFormat);
    expect($dataFormat).toEqual('channelsFirst');
  });

  it('throws unknown dataFormat', () => {
    const dataFormat = 'FakeFormat';
    expect(
        () => conv_util.convertConv2DDataFormat(dataFormat as 'NHWC' | 'NCHW'))
        .toThrowError();
  });
});
