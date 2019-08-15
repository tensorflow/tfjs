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

export let executeOp: InternalOpExecutor = (node: Node,
                                            tensorMap: NamedTensorsMap,
                                            context: ExecutionContext):
                                               tfc.Tensor[] => {
  switch (node.op) {
    case 'Fill': {
      const shape =
          getParamValue('shape', node, tensorMap, context) as number[];
      const dtype =
          getParamValue('dtype', node, tensorMap, context) as tfc.DataType;
      const value = getParamValue('value', node, tensorMap, context) as number;
      return [tfc.fill(shape, value, dtype)];
    }
    case 'LinSpace': {
      const start = getParamValue('start', node, tensorMap, context) as number;
      const stop = getParamValue('stop', node, tensorMap, context) as number;
      const num = getParamValue('num', node, tensorMap, context) as number;
      return [tfc.linspace(start, stop, num)];
    }
    case 'OneHot': {
      const indices =
          getParamValue('indices', node, tensorMap, context) as tfc.Tensor1D;
      const depth = getParamValue('depth', node, tensorMap, context) as number;
      const onValue =
          getParamValue('onValue', node, tensorMap, context) as number;
      const offValue =
          getParamValue('offValue', node, tensorMap, context) as number;
      return [tfc.oneHot(indices, depth, onValue, offValue)];
    }
    case 'Ones': {
      return [tfc.ones(
          getParamValue('shape', node, tensorMap, context) as number[],
          getParamValue('dtype', node, tensorMap, context) as tfc.DataType)];
    }
    case 'OnesLike': {
      return [tfc.onesLike(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'RandomUniform': {
      return [tfc.randomUniform(
          // tslint:disable-next-line:no-any
          getParamValue('shape', node, tensorMap, context) as any,
          getParamValue('minval', node, tensorMap, context) as number,
          getParamValue('maxval', node, tensorMap, context) as number,
          getParamValue('dtype', node, tensorMap, context) as tfc.DataType)];
    }
    case 'Range': {
      const start = getParamValue('start', node, tensorMap, context) as number;
      const stop = getParamValue('stop', node, tensorMap, context) as number;
      const step = getParamValue('step', node, tensorMap, context) as number;
      return [tfc.range(
          start, stop, step,
          getParamValue('dtype', node, tensorMap, context) as 'float32' |
              'int32')];
    }
    case 'TruncatedNormal': {
      const shape =
          getParamValue('shape', node, tensorMap, context) as number[];
      const mean = getParamValue('mean', node, tensorMap, context) as number;
      const stdDev =
          getParamValue('stdDev', node, tensorMap, context) as number;
      const seed = getParamValue('seed', node, tensorMap, context) as number;
      return [tfc.truncatedNormal(
          shape, mean, stdDev,
          getParamValue('dtype', node, tensorMap, context) as 'float32' |
              'int32',
          seed)];
    }
    case 'Zeros': {
      return [tfc.zeros(
          getParamValue('shape', node, tensorMap, context) as number[],
          getParamValue('dtype', node, tensorMap, context) as tfc.DataType)];
    }
    case 'ZerosLike': {
      return [tfc.zerosLike(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'creation';
