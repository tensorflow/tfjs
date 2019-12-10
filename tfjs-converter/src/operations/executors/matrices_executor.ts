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
    case 'BatchMatMul':
    case 'BatchMatMulV2':
    case 'MatMul':
      return [tfc.matMul(
          getParamValue('a', node, tensorMap, context) as tfc.Tensor2D,
          getParamValue('b', node, tensorMap, context) as tfc.Tensor2D,
          getParamValue('transposeA', node, tensorMap, context) as boolean,
          getParamValue('transposeB', node, tensorMap, context) as boolean)];

    case 'Transpose':
      return [tfc.transpose(
          getParamValue('x', node, tensorMap, context) as tfc.Tensor,
          getParamValue('perm', node, tensorMap, context) as number[])];

    case '_FusedMatMul':
      const [extraOp, activationFunc] =
          (getParamValue('fusedOps', node, tensorMap, context) as string[]);

      const isBiasAdd = extraOp === 'biasadd';
      const isPrelu = activationFunc === 'prelu';

      const numArgs =
          (getParamValue('numArgs', node, tensorMap, context) as number);
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
          getParamValue('args', node, tensorMap, context) as tfc.Tensor[];
      return [tfc.fused.matMul({
        a: getParamValue('a', node, tensorMap, context) as tfc.Tensor2D,
        b: getParamValue('b', node, tensorMap, context) as tfc.Tensor2D,
        transposeA: getParamValue('transposeA', node, tensorMap, context) as
            boolean,
        transposeB: getParamValue('transposeB', node, tensorMap, context) as
            boolean,
        bias: biasArg,
        activation: activationFunc as tfc.fused.Activation,
        preluActivationWeights: preluArg
      })];

    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'matrices';
