/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import * as dl from 'deeplearn';
import {DataType} from 'deeplearn/dist/types';

import {NamedTensorMap} from '../../data/index';
import {Node} from '../index';

import {OpExecutor} from './types';
import {getParamValue} from './utils';

export let executeOp: OpExecutor = (node: Node,
                                    tensorMap: NamedTensorMap): dl.Tensor => {
  switch (node.op) {
    case 'fill': {
      const shape = getParamValue('shape', node, tensorMap) as number[];
      const value = getParamValue('value', node, tensorMap) as number;
      return dl.fill(shape, value);
    }
    case 'linspace': {
      const start = getParamValue('start', node, tensorMap) as number;
      const stop = getParamValue('stop', node, tensorMap) as number;
      const num = getParamValue('num', node, tensorMap) as number;
      return dl.linspace(start, stop, num);
    }
    case 'oneHot': {
      const indices = getParamValue('indices', node, tensorMap) as dl.Tensor1D;
      const depth = getParamValue('depth', node, tensorMap) as number;
      const onValue = getParamValue('onValue', node, tensorMap) as number;
      const offValue = getParamValue('offValue', node, tensorMap) as number;
      return dl.oneHot(indices, depth, onValue, offValue);
    }
    case 'ones': {
      return dl.ones(
          getParamValue('shape', node, tensorMap) as number[],
          getParamValue('dtype', node, tensorMap) as DataType);
    }
    case 'onesLike': {
      return dl.onesLike(getParamValue('x', node, tensorMap) as dl.Tensor);
    }
    case 'randomUniform': {
      return dl.randomUniform(
          // tslint:disable-next-line:no-any
          getParamValue('shape', node, tensorMap) as any,
          getParamValue('minval', node, tensorMap) as number,
          getParamValue('maxval', node, tensorMap) as number,
          getParamValue('dtype', node, tensorMap) as DataType);
    }
    case 'range': {
      const start = getParamValue('start', node, tensorMap) as number;
      const stop = getParamValue('stop', node, tensorMap) as number;
      const step = getParamValue('step', node, tensorMap) as number;
      return dl.range(
          start, stop, step,
          getParamValue('dtype', node, tensorMap) as 'float32' | 'int32');
    }
    case 'truncatedNormal': {
      const shape = getParamValue('shape', node, tensorMap) as number[];
      const mean = getParamValue('mean', node, tensorMap) as number;
      const stdDev = getParamValue('stdDev', node, tensorMap) as number;
      const seed = getParamValue('seed', node, tensorMap) as number;
      return dl.truncatedNormal(
          shape, mean, stdDev,
          getParamValue('dtype', node, tensorMap) as 'float32' | 'int32', seed);
    }
    case 'zeros': {
      return dl.zeros(
          getParamValue('shape', node, tensorMap) as number[],
          getParamValue('dtype', node, tensorMap) as DataType);
    }
    case 'zerosLike': {
      return dl.zerosLike(getParamValue('x', node, tensorMap) as dl.Tensor);
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'creation';
