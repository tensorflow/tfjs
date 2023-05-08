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

import {getParamValue, getTensor} from './utils';

export const executeOp: InternalOpExecutor =
    (node: Node, tensorMap: NamedTensorsMap, context: ExecutionContext,
     ops = tfOps): Tensor[] => {
      switch (node.op) {
        case 'Abs':
        case 'ComplexAbs':
          return [ops.abs(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acos':
          return [ops.acos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acosh':
          return [ops.acosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asin':
          return [ops.asin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asinh':
          return [ops.asinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan':
          return [ops.atan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan2':
          return [ops.atan2(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('y', node, tensorMap, context) as Tensor)];
        case 'Atanh':
          return [ops.atanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Ceil':
          return [ops.ceil(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Complex':
          return [ops.complex(
              getParamValue('real', node, tensorMap, context) as Tensor,
              getParamValue('imag', node, tensorMap, context) as Tensor)];
        case 'Cos':
          return [ops.cos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Cosh':
          return [ops.cosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Elu':
          return [ops.elu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Erf':
          return [ops.erf(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Exp':
          return [ops.exp(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Expm1': {
          return [ops.expm1(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Floor':
          return [ops.floor(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log':
          return [ops.log(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log1p': {
          return [ops.log1p(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Imag':
          return [ops.imag(
              getParamValue('x', node, tensorMap, context) as Tensor)];

        case 'Neg':
          return [ops.neg(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Reciprocal': {
          return [ops.reciprocal(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Real':
          return [ops.real(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Relu':
          return [ops.relu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Round': {
          return [ops.round(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Selu':
          return [ops.selu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sigmoid':
          return [ops.sigmoid(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sin':
          return [ops.sin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sign': {
          return [ops.sign(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sinh': {
          return [ops.sinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Softplus': {
          return [ops.softplus(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sqrt': {
          return [ops.sqrt(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Square': {
          return [ops.square(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tanh': {
          return [ops.tanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tan':
          return [ops.tan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'ClipByValue':
          return [ops.clipByValue(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('clipValueMin', node, tensorMap, context) as number,
              getParamValue('clipValueMax', node, tensorMap, context) as
                  number)];
        case 'Relu6':
          return [ops.relu6(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Rsqrt':
          return [ops.rsqrt(getTensor(node.inputNames[0], tensorMap, context))];
        case 'LeakyRelu':
          return [ops.leakyRelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as number)];
        case 'Prelu':
          return [ops.prelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as Tensor)];
        case 'IsNan':
          return [ops.isNaN(getTensor(node.inputNames[0], tensorMap, context))];
        case 'IsInf':
          return [ops.isInf(getTensor(node.inputNames[0], tensorMap, context))];
        case 'IsFinite':
          return [ops.isFinite(
              getTensor(node.inputNames[0], tensorMap, context))];
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'basic_math';
