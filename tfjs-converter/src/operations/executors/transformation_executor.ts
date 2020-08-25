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

import {Tensor, Tensor4D} from '@tensorflow/tfjs-core';
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): Tensor[] => {
      switch (node.op) {
        case 'Cast': {
          return [tfOps.cast(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('dtype', node, tensorMap, context) as 'int32' |
                  'float32' | 'bool')];
        }
        case 'ExpandDims': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number;
          return [tfOps.expandDims(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }
        case 'Squeeze': {
          const axis =
              getParamValue('axis', node, tensorMap, context) as number[];
          return [tfOps.squeeze(
              getParamValue('x', node, tensorMap, context) as Tensor, axis)];
        }

        case 'Reshape': {
          return [tfOps.reshape(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as number[])];
        }
        case 'PadV2':
        case 'Pad': {
          return [tfOps.pad(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('padding', node, tensorMap, context) as
                  Array<[number, number]>,
              getParamValue('constantValue', node, tensorMap, context) as
                  number)];
        }
        case 'SpaceToBatchND': {
          const blockShape =
              getParamValue('blockShape', node, tensorMap, context) as number[];
          const paddings =
              getParamValue('paddings', node, tensorMap, context) as number[][];
          return [tfOps.spaceToBatchND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape, paddings)];
        }
        case 'BatchToSpaceND': {
          const blockShape =
              getParamValue('blockShape', node, tensorMap, context) as number[];
          const crops =
              getParamValue('crops', node, tensorMap, context) as number[][];
          return [tfOps.batchToSpaceND(
              getParamValue('x', node, tensorMap, context) as Tensor,
              blockShape, crops)];
        }
        case 'DepthToSpace': {
          const blockSize =
              getParamValue('blockSize', node, tensorMap, context) as number;
          const dataFormat =
              (getParamValue('dataFormat', node, tensorMap, context) as
               string).toUpperCase() as 'NHWC' |
              'NCHW';
          return [tfOps.depthToSpace(
              getParamValue('x', node, tensorMap, context) as Tensor4D,
              blockSize, dataFormat)];
        }
        case 'BroadcastTo': {
          return [tfOps.broadcastTo(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('shape', node, tensorMap, context) as number[])];
        }
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'transformation';
