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

import {Tensor, Tensor2D} from '@tensorflow/tfjs-core';
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
        case 'BatchMatMul':
        case 'BatchMatMulV2':
        case 'MatMul':
          return [tfOps.matMul(
              getParamValue('a', node, tensorMap, context) as Tensor2D,
              getParamValue('b', node, tensorMap, context) as Tensor2D,
              getParamValue('transposeA', node, tensorMap, context) as boolean,
              getParamValue('transposeB', node, tensorMap, context) as
                  boolean)];

        case 'Einsum':
          return [tfOps.einsum(
              getParamValue('equation', node, tensorMap, context) as string,
              ...getParamValue('tensors', node, tensorMap, context) as
                  Tensor[])];

        case 'Transpose':
          return [tfOps.transpose(
              getParamValue('x', node, tensorMap, context) as Tensor,
              getParamValue('perm', node, tensorMap, context) as number[])];

        case '_FusedMatMul':
          const [extraOp, activationFunc] =
              (getParamValue('fusedOps', node, tensorMap, context) as string[]);

          const isBiasAdd = extraOp === 'biasadd';
          const isPrelu = activationFunc === 'prelu';

          const numArgs =
              (getParamValue('numArgs', node, tensorMap, context) as number);
          const leakyreluAlpha =
              getParamValue('leakyreluAlpha', node, tensorMap, context) as
              number;

          if (isBiasAdd) {
            if (isPrelu && numArgs !== 2) {
              throw new Error(
                  'Fused MatMul with BiasAdd and Prelu must have two ' +
                  'extra arguments: bias and alpha.');
            }
            if (!isPrelu && numArgs !== 1) {
              throw new Error(
                  'Fused MatMul with BiasAdd must have one extra argument: bias.');
            }
          }
          const [biasArg, preluArg] =
              getParamValue('args', node, tensorMap, context) as Tensor[];
          return [tfOps.fused.matMul({
            a: getParamValue('a', node, tensorMap, context) as Tensor2D,
            b: getParamValue('b', node, tensorMap, context) as Tensor2D,
            transposeA: getParamValue('transposeA', node, tensorMap, context) as
                boolean,
            transposeB: getParamValue('transposeB', node, tensorMap, context) as
                boolean,
            bias: biasArg,
            activation: activationFunc as tfOps.fused.Activation,
            preluActivationWeights: preluArg,
            leakyreluAlpha
          })];

        default:
          throw TypeError(`Node type ${node.op} is not implemented`);
      }
    };

export const CATEGORY = 'matrices';
