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
import {getParamValue, getTensor} from './utils';

export let executeOp: OpExecutor = (node: Node, tensorMap: NamedTensorsMap,
                                    context: ExecutionContext):
                                       tfc.Tensor[] => {
  switch (node.op) {
    case 'abs':
      return [tfc.abs(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'acos':
      return [tfc.acos(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'acosh':
      return [tfc.acosh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'asin':
      return [tfc.asin(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'asinh':
      return [tfc.asinh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'atan':
      return [tfc.atan(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'atanh':
      return [tfc.atanh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'ceil':
      return [tfc.ceil(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'cos':
      return [tfc.cos(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'cosh':
      return [tfc.cosh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'elu':
      return [tfc.elu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'erf':
      return [tfc.erf(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'exp':
      return [tfc.exp(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'expm1': {
      return [tfc.expm1(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'floor':
      return [tfc.floor(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'log':
      return [tfc.log(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'log1p': {
      return [tfc.log1p(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'neg':
      return [tfc.neg(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'reciprocal': {
      return [tfc.reciprocal(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'relu':
      return [tfc.relu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'round': {
      return [tfc.round(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'selu':
      return [tfc.selu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'sigmoid':
      return [tfc.sigmoid(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'sin':
      return [tfc.sin(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'sign': {
      return [tfc.sign(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'sinh': {
      return [tfc.sinh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'softplus': {
      return [tfc.softplus(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'sqrt': {
      return [tfc.sqrt(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'square': {
      return [tfc.square(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'tanh': {
      return [tfc.tanh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'tan':
      return [tfc.tan(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'clipByValue':
      return [tfc.clipByValue(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('clipValueMin', node, tensorMap, context) as number,
          getParamValue('clipValueMax', node, tensorMap, context) as number)];
    case 'rsqrt':
      return [tfc.div(
          tfc.scalar(1.0, 'float32'),
          tfc.sqrt(getTensor(node.inputNames[0], tensorMap, context)))];

    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'basic_math';
