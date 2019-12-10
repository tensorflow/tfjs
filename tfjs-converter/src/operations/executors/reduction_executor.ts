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
import {InternalOpExecutor, Node} from '../types';

import {getParamValue} from './utils';

export const executeOp: InternalOpExecutor = (node: Node,
                                            tensorMap: NamedTensorsMap,
                                            context: ExecutionContext):
                                               tfc.Tensor[] => {
  switch (node.op) {
    case 'Max': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.max(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'Mean': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.mean(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'Min': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.min(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'Sum': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.sum(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'All': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.all(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'Any': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.any(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    case 'ArgMax': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      return [tfc.argMax(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis)];
    }
    case 'ArgMin': {
      const axis = getParamValue('axis', node, tensorMap, context) as number;
      return [tfc.argMin(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis)];
    }
    case 'Prod': {
      const axis = getParamValue('axis', node, tensorMap, context) as number[];
      const keepDims =
          getParamValue('keepDims', node, tensorMap, context) as boolean;
      return [tfc.prod(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor, axis,
          keepDims)];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'reduction';
