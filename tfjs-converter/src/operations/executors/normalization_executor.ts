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

import {Tensor, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext, ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'EuclideanNorm':
          return [ops.euclideanNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('axis', node, tensorMap, context) as number[],
              getParamValue('keepDims', node, tensorMap, context) as boolean)];
        case 'FusedBatchNorm':
        case 'FusedBatchNormV2': {
          return [ops.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              getParamValue('offset', node, tensorMap, context) as Tensor,
              getParamValue('scale', node, tensorMap, context) as Tensor,
              getParamValue('epsilon', node, tensorMap, context) as number)];
        }
        case 'FusedBatchNormV3': {
          return [ops.batchNorm(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('mean', node, tensorMap, context) as Tensor,
              getParamValue('variance', node, tensorMap, context) as Tensor,
              getParamValue('offset', node, tensorMap, context) as Tensor,
              getParamValue('scale', node, tensorMap, context) as Tensor,
              getParamValue('epsilon', node, tensorMap, context) as number)];
        }
        case 'LRN': {
          return [ops.localResponseNormalization(
              getParamValue('x', node, tensorMap, context) as Tensor3D |
                  Tensor4D,
              getParamValue('radius', node, tensorMap, context) as number,
              getParamValue('bias', node, tensorMap, context) as number,
              getParamValue('alpha', node, tensorMap, context) as number,
              getParamValue('beta', node, tensorMap, context) as number)];
        }
        case 'Softmax': {
          return [ops.softmax(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'LogSoftmax': {
          return [ops.logSoftmax(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'normalization';
