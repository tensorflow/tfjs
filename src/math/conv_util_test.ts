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

import * as conv_util from './conv_util';

describe('conv_util computeConvInfo', () => {
  it('1x1 conv over 1x1 array with same pad', () => {
    const inShape: [number, number, number] = [1, 1, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 1, 1, 1, 1, 1, 'same');
    expect(convInfo.outShape).toEqual([1, 1, 1]);
  });

  it('2x2 conv over 3x3 array with same pad', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'same');
    expect(convInfo.outShape).toEqual([3, 3, 1]);
    // Should produce non-even padding with extra pixel at the right/bottom.
    expect(convInfo.padInfo.left).toBe(0);
    expect(convInfo.padInfo.right).toBe(1);
    expect(convInfo.padInfo.top).toBe(0);
    expect(convInfo.padInfo.bottom).toBe(1);
  });

  it('2x2 conv over 3x3 array with same pad', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'same');
    expect(convInfo.outShape).toEqual([3, 3, 1]);
  });

  it('2x2 conv over 3x3 array with valid pad', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'valid');
    expect(convInfo.outShape).toEqual([2, 2, 1]);
  });

  it('2x2 conv over 3x3 array with valid pad with stride 2', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 2, 2, 'valid');
    expect(convInfo.outShape).toEqual([1, 1, 1]);
  });

  it('2x2 conv over 3x3 array with valid pad with stride 2', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 2, 2, 'valid');
    expect(convInfo.outShape).toEqual([1, 1, 1]);
  });

  it('2x1 conv over 3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 1, 1, 1, 1, 'valid');
    expect(convInfo.outShape).toEqual([2, 3, 1]);
  });

  it('2x1 conv over 3x3 array with valid pad with strides h=2, w=1', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 2, 1, 1, 2, 1, 'valid');
    expect(convInfo.outShape).toEqual([1, 3, 1]);
  });

  it('1x2 conv over 3x3 array with valid pad with stride 1', () => {
    const inShape: [number, number, number] = [3, 3, 1];
    const convInfo = conv_util.computeConvInfo(inShape, 1, 2, 1, 1, 1, 'valid');
    expect(convInfo.outShape).toEqual([3, 2, 1]);
  });
});
