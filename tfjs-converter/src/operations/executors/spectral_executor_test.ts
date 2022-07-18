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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {ExecutionContext} from '../../executor/execution_context';
import * as spectral from '../op_list/spectral';
import {Node} from '../types';

import {executeOp} from './spectral_executor';
import {createTensorAttr, validateParam} from './test_helper';
import {RecursiveSpy, spyOnAllFunctions} from './spy_ops';

describe('spectral', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});
  let spyOps: RecursiveSpy<typeof tfOps>;
  let spyOpsAsTfOps: typeof tfOps;

  beforeEach(() => {
    spyOps = spyOnAllFunctions(tfOps);
    spyOpsAsTfOps = spyOps as unknown as typeof tfOps;
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
      it('should call tfOps.fft', () => {
        node.op = 'FFT';
        spyOps.fft.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.fft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'FFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('IFFT', () => {
      it('should call tfOps.ifft', () => {
        node.op = 'IFFT';
        spyOps.ifft.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.ifft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'IFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('RFFT', () => {
      it('should call tfOps.rfft', () => {
        node.op = 'RFFT';
        spyOps.rfft.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.rfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'RFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
    describe('IRFFT', () => {
      it('should call tfOps.irfft', () => {
        node.op = 'IRFFT';
        spyOps.irfft.and.returnValue({});
        executeOp(node, {input1}, context, spyOpsAsTfOps);

        expect(spyOps.irfft).toHaveBeenCalledWith(input1[0]);
      });
      it('should match json def', () => {
        node.op = 'IRFFT';

        expect(validateParam(node, spectral.json)).toBeTruthy();
      });
    });
  });
});
