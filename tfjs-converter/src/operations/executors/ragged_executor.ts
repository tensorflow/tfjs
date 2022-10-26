/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {Tensor, Tensor1D} from '@tensorflow/tfjs-core';
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
        case 'RaggedGather': {
          const {
            outputNestedSplits,
            outputDenseValues,
          } =
              ops.raggedGather(
                  getParamValue(
                      'paramsNestedSplits', node, tensorMap, context) as
                      Tensor[],
                  getParamValue(
                      'paramsDenseValues', node, tensorMap, context) as Tensor,
                  getParamValue('indices', node, tensorMap, context) as Tensor,
                  getParamValue('outputRaggedRank', node, tensorMap, context) as
                      number);
          return outputNestedSplits.concat(outputDenseValues);
        }
        case 'RaggedRange': {
          const {rtNestedSplits, rtDenseValues} = ops.raggedRange(
              getParamValue('starts', node, tensorMap, context) as Tensor,
              getParamValue('limits', node, tensorMap, context) as Tensor,
              getParamValue('splits', node, tensorMap, context) as Tensor);
          return [rtNestedSplits, rtDenseValues];
        }
        case 'RaggedTensorToTensor': {
          return [ops.raggedTensorToTensor(
              getParamValue('shape', node, tensorMap, context) as Tensor,
              getParamValue('values', node, tensorMap, context) as Tensor1D,
              getParamValue('defaultValue', node, tensorMap, context) as Tensor,
              getParamValue('rowPartitionTensors', node, tensorMap, context) as
                  Tensor[],
              getParamValue('rowPartitionTypes', node, tensorMap, context) as
                  string[])];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'ragged';
