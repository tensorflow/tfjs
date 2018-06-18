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
import {getParamValue} from './utils';

export let executeOp: OpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): tfc.Tensor[] => {
      switch (node.op) {
        case 'add': {
          return [tfc.add(
              (getParamValue('a', node, tensorMap, context) as tfc.Tensor),
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'mod':
          return [tfc.mod(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        case 'mul':
          return [tfc.mul(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        case 'div': {
          return [tfc.div(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'floorDiv': {
          return [tfc.floorDiv(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'sub': {
          return [tfc.sub(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'minimum': {
          return [tfc.minimum(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'maximum': {
          return [tfc.maximum(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'pow': {
          return [tfc.pow(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        case 'squaredDifference': {
          return [tfc.squaredDifference(
              getParamValue('a', node, tensorMap, context) as tfc.Tensor,
              getParamValue('b', node, tensorMap, context) as tfc.Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'arithmetic';
