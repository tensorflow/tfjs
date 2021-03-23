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
    (node: Node, tensorMap: NamedTensorsMap,
     context: ExecutionContext): Tensor[] => {
      switch (node.op) {
        case 'Abs':
        case 'ComplexAbs':
          return [tfOps.abs(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acos':
          return [tfOps.acos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Acosh':
          return [tfOps.acosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asin':
          return [tfOps.asin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Asinh':
          return [tfOps.asinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan':
          return [tfOps.atan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Atan2':
          return [tfOps.atan2(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('y', node, tensorMap, context) as Tensor)];
        case 'Atanh':
          return [tfOps.atanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Ceil':
          return [tfOps.ceil(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Complex':
          return [tfOps.complex(
              getParamValue('real', node, tensorMap, context) as Tensor,
              getParamValue('imag', node, tensorMap, context) as Tensor)];
        case 'Cos':
          return [tfOps.cos(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Cosh':
          return [tfOps.cosh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Elu':
          return [tfOps.elu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Erf':
          return [tfOps.erf(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Exp':
          return [tfOps.exp(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Expm1': {
          return [tfOps.expm1(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Floor':
          return [tfOps.floor(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log':
          return [tfOps.log(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Log1p': {
          return [tfOps.log1p(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Imag':
          return [tfOps.imag(
              getParamValue('x', node, tensorMap, context) as Tensor)];

        case 'Neg':
          return [tfOps.neg(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Reciprocal': {
          return [tfOps.reciprocal(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Real':
          return [tfOps.real(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Relu':
          return [tfOps.relu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Round': {
          return [tfOps.round(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Selu':
          return [tfOps.selu(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sigmoid':
          return [tfOps.sigmoid(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sin':
          return [tfOps.sin(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Sign': {
          return [tfOps.sign(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sinh': {
          return [tfOps.sinh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Softplus': {
          return [tfOps.softplus(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Sqrt': {
          return [tfOps.sqrt(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Square': {
          return [tfOps.square(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tanh': {
          return [tfOps.tanh(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        }
        case 'Tan':
          return [tfOps.tan(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'ClipByValue':
          return [tfOps.clipByValue(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('clipValueMin', node, tensorMap, context) as number,
              getParamValue('clipValueMax', node, tensorMap, context) as
                  number)];
        case 'Relu6':
          return [tfOps.relu6(
              getParamValue('x', node, tensorMap, context) as Tensor)];
        case 'Rsqrt':
          return [tfOps.rsqrt(
              getTensor(node.inputNames[0], tensorMap, context))];
        case 'Prod':
          return [tfOps.prod(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('axes', node, tensorMap, context) as number[])];
        case 'LeakyRelu':
          return [tfOps.leakyRelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as number)];
        case 'Prelu':
          return [tfOps.prelu(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('alpha', node, tensorMap, context) as Tensor)];
        case 'IsNan':
          return [tfOps.isNaN(
              getTensor(node.inputNames[0], tensorMap, context))];
        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'basic_math';
