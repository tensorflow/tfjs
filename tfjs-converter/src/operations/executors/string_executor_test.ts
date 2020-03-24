/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import {Node} from '../types';

import {executeOp} from './string_executor';
import {createBoolAttr, createTensorAttr} from './test_helper';

describe('string', () => {
  let node: Node;
  const input1 = [tfc.tensor(['a'], [1], 'string')];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'string',
      inputNames: ['input1'],
      inputs: [],
      inputParams: {str: createTensorAttr(0)},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('DecodeBase64', () => {
      it('should call tfc.decodeBase64', () => {
        spyOn(tfc, 'decodeBase64');
        node.op = 'DecodeBase64';
        executeOp(node, {input1}, context);
        expect(tfc.decodeBase64).toHaveBeenCalledWith(input1[0]);
      });
    });
    describe('EncodeBase64', () => {
      it('should call tfc.encodeBase64', () => {
        spyOn(tfc, 'encodeBase64');
        node.op = 'EncodeBase64';
        node.attrParams.pad = createBoolAttr(true);
        executeOp(node, {input1}, context);
        expect(tfc.encodeBase64).toHaveBeenCalledWith(input1[0], true);
      });
    });
  });
});
