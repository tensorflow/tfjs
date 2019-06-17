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

import {getParamValue, getTensor} from './utils';

export let executeOp: InternalOpExecutor = (node: Node,
                                            tensorMap: NamedTensorsMap,
                                            context: ExecutionContext):
                                               tfc.Tensor[] => {
  switch (node.op) {
    case 'Abs':
    case 'ComplexAbs':
      return [tfc.abs(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Acos':
      return [tfc.acos(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Acosh':
      return [tfc.acosh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Asin':
      return [tfc.asin(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Asinh':
      return [tfc.asinh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Atan':
      return [tfc.atan(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Atan2':
      return [tfc.atan2(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('y', node, tensorMap, context) as tfc.Tensor)];
    case 'Atanh':
      return [tfc.atanh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Ceil':
      return [tfc.ceil(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Complex':
      return [tfc.complex(
          getParamValue('real', node, tensorMap, context) as tfc.Tensor,
          getParamValue('imag', node, tensorMap, context) as tfc.Tensor)];
    case 'Cos':
      return [tfc.cos(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Cosh':
      return [tfc.cosh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Elu':
      return [tfc.elu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Erf':
      return [tfc.erf(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Exp':
      return [tfc.exp(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Expm1': {
      return [tfc.expm1(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Floor':
      return [tfc.floor(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Log':
      return [tfc.log(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Log1p': {
      return [tfc.log1p(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Neg':
      return [tfc.neg(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Reciprocal': {
      return [tfc.reciprocal(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Relu':
      return [tfc.relu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Round': {
      return [tfc.round(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Selu':
      return [tfc.selu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Sigmoid':
      return [tfc.sigmoid(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Sin':
      return [tfc.sin(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Sign': {
      return [tfc.sign(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Sinh': {
      return [tfc.sinh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Softplus': {
      return [tfc.softplus(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Sqrt': {
      return [tfc.sqrt(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Square': {
      return [tfc.square(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Tanh': {
      return [tfc.tanh(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    }
    case 'Tan':
      return [tfc.tan(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor)];
    case 'Relu6':
    case 'ClipByValue':
      return [tfc.clipByValue(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('clipValueMin', node, tensorMap, context) as number,
          getParamValue('clipValueMax', node, tensorMap, context) as number)];
    case 'Rsqrt':
      return [tfc.rsqrt(getTensor(node.inputNames[0], tensorMap, context))];
    case 'Prod':
      return [tfc.prod(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('axes', node, tensorMap, context) as number[])];
    case 'LeakyRelu':
      return [tfc.leakyRelu(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('alpha', node, tensorMap, context) as number)];
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'basic_math';
