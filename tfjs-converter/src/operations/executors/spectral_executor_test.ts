/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as tfc from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../../executor/execution_context';
import * as spectral from '../op_list/spectral';
import {Node} from '../types';

import {executeOp} from './spectral_executor';
import {createTensorAttr, validateParam} from './test_helper';

describe('spectral', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'spectral',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('FFT', () => {
      it('should call tfc.fft', () => {
        spyOn(tfc, 'fft');
        node.op = 'FFT';
        executeOp(node, {input1}, context);

        expect(tfc.fft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'FFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('IFFT', () => {
      it('should call tfc.ifft', () => {
        spyOn(tfc, 'ifft');
        node.op = 'IFFT';
        executeOp(node, {input1}, context);

        expect(tfc.ifft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'IFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('RFFT', () => {
      it('should call tfc.rfft', () => {
        spyOn(tfc, 'rfft');
        node.op = 'RFFT';
        executeOp(node, {input1}, context);

        expect(tfc.rfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'RFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('IRFFT', () => {
      it('should call tfc.irfft', () => {
        spyOn(tfc, 'irfft');
        node.op = 'IRFFT';
        executeOp(node, {input1}, context);

        expect(tfc.irfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should reshape result for 3d', () => {
        const result = tfc.tensor2d([2, 2, 2, 2], [2, 2]);
        const input2 = [tfc.tensor3d([2, 2, 2, 2], [1, 2, 2])];
        spyOn(tfc, 'irfft').and.returnValue(result);
        node.op = 'IRFFT';
        node.inputNames = ['input2'];
        const output = executeOp(node, {input2}, context) as tfc.Tensor[];
        expect(output[0].rank).toEqual(3);
        expect(output[0].shape).toEqual([1, 2, 2]);
      });
      it('should match json def', () => {
        node.op = 'IRFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
  });
});
