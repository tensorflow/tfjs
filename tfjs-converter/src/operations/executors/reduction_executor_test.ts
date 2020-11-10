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
import {Node} from '../types';

import {executeOp} from './reduction_executor';
import {createBoolAttr, createNumberAttr, createNumberAttrFromIndex, createTensorAttr} from './test_helper';

describe('reduction', () => {
  let node: Node;
  const input1 = [tfOps.scalar(1)];
  const context = new ExecutionContext({}, {}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'logical',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {x: createTensorAttr(0)},
      attrParams: {keepDims: createBoolAttr(true), axis: createNumberAttr(1)},
      children: []
    };
  });

  describe('executeOp', () => {
    ['Max', 'Mean', 'Min', 'Sum', 'All', 'Any', 'Prod'].forEach(op => {
      it('should call tfOps.' + op, () => {
        const spy =
            spyOn(tfOps, op.charAt(0).toLowerCase() + op.slice(1) as 'max');
        node.op = op;
        executeOp(node, {input1}, context);

        expect(spy).toHaveBeenCalledWith(input1[0], 1, true);
      });
    });
    describe('ArgMax', () => {
      it('should call tfOps.argMax', () => {
        spyOn(tfOps, 'argMax');
        node.op = 'ArgMax';
        executeOp(node, {input1}, context);

        expect(tfOps.argMax).toHaveBeenCalledWith(input1[0], 1);
      });
    });
    describe('ArgMin', () => {
      it('should call tfOps.argMin', () => {
        spyOn(tfOps, 'argMin');
        node.op = 'ArgMin';
        executeOp(node, {input1}, context);

        expect(tfOps.argMin).toHaveBeenCalledWith(input1[0], 1);
      });
      describe('Cumsum', () => {
        it('should call tfOps.cumsum', () => {
          spyOn(tfOps, 'cumsum');
          node.op = 'Cumsum';
          node.attrParams.exclusive = createBoolAttr(true);
          node.attrParams.reverse = createBoolAttr(false);
          node.inputNames = ['input1', 'input2'];
          node.inputParams.axis = createNumberAttrFromIndex(1);
          const input2 = [tfOps.scalar(2)];
          executeOp(node, {input1, input2}, context);

          expect(tfOps.cumsum).toHaveBeenCalledWith(input1[0], 2, true, false);
        });
      });
    });
  });
});
