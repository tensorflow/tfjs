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

import {Tensor} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Equal': {
          return [ops.equal(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'NotEqual': {
          return [ops.notEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Greater': {
          return [ops.greater(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'GreaterEqual': {
          return [ops.greaterEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Less': {
          return [ops.less(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LessEqual': {
          return [ops.lessEqual(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalAnd': {
          return [ops.logicalAnd(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalNot': {
          return [ops.logicalNot(
              getParamValue('a', node, tensorMap, context) as Tensor)];
        }
        case 'LogicalOr': {
          return [ops.logicalOr(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'Select':
        case 'SelectV2': {
          return [ops.where(
              getParamValue('condition', node, tensorMap, context) as Tensor,
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        case 'BitwiseAnd': {
          return [ops.bitwiseAnd(
              getParamValue('a', node, tensorMap, context) as Tensor,
              getParamValue('b', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'logical';
