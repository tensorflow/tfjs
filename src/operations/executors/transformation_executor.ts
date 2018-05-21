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

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {Node} from '../types';

import {OpExecutor} from './types';
import {getParamValue, split} from './utils';

export let executeOp: OpExecutor = (node: Node, tensorMap: NamedTensorsMap,
                                    context: ExecutionContext):
                                       tfc.Tensor[] => {
  switch (node.op) {
    case 'cast': {
      return [tfc.cast(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('dtype', node, tensorMap, context) as 'int32' |
              'float32' | 'bool')];
    }
    case 'expandDims': {
      const axis = node.params['axis'].value as number;
      return [tfc.expandDims(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis)];
    }
    case 'squeeze': {
      const axis = node.params['axis'].value as number[];
      return [tfc.squeeze(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis)];
    }

    case 'reshape': {
      return [tfc.reshape(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('shape', node, tensorMap, context) as number[])];
    }
    case 'pad': {
      return [tfc.pad(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          split(
              getParamValue('padding', node, tensorMap, context) as number[],
              2) as Array<[number, number]>,
          getParamValue('constantValue', node, tensorMap, context) as number)];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'transformation';
