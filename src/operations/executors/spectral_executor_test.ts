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
import {Node, OpMapper} from '../types';

import {executeOp} from './spectral_executor';
import {createTensorAttr, validateParam} from './test_helper';

describe('spectral', () => {
  let node: Node;
  const input1 = [tfc.scalar(1)];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'spectral',
      inputNames: ['input1'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('fft', () => {
      it('should call tfc.fft', () => {
        spyOn(tfc, 'fft');
        node.op = 'fft';
        executeOp(node, {input1}, context);

        expect(tfc.fft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'fft';

        expect(validateParam(node, spectral.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('ifft', () => {
      it('should call tfc.ifft', () => {
        spyOn(tfc, 'ifft');
        node.op = 'ifft';
        executeOp(node, {input1}, context);

        expect(tfc.ifft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'ifft';

        expect(validateParam(node, spectral.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('rfft', () => {
      it('should call tfc.rfft', () => {
        spyOn(tfc, 'rfft');
        node.op = 'rfft';
        executeOp(node, {input1}, context);

        expect(tfc.rfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'rfft';

        expect(validateParam(node, spectral.json as OpMapper[])).toBeTruthy();
      });
    });
    describe('irfft', () => {
      it('should call tfc.irfft', () => {
        spyOn(tfc, 'irfft');
        node.op = 'irfft';
        executeOp(node, {input1}, context);

        expect(tfc.irfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'irfft';

        expect(validateParam(node, spectral.json as OpMapper[])).toBeTruthy();
      });
    });
  });
});
