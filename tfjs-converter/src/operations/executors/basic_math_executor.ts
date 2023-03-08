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

import {ExecutionContext} from '../../executor/execution_context';
import {InternalOpExecutor} from '../types';
import type {OpExecutorBuilder} from '../operation_executor';

export function buildOpExecutor(builder: OpExecutorBuilder):
    InternalOpExecutor {
  const node = builder.node;
  const ops = builder.manager.ops;

  function unaryOpExecutor(op: typeof ops.abs): InternalOpExecutor {
    const x$ = builder.param('x');
    return (ctx: ExecutionContext) => [op(ctx.getOpParamValue(x$))];
  }

  switch (node.op) {
    case 'Abs':
    case 'ComplexAbs':
      return unaryOpExecutor(ops.abs);
    case 'Acos':
      return unaryOpExecutor(ops.acos);
    case 'Acosh':
      return unaryOpExecutor(ops.acosh);
    case 'Asin':
      return unaryOpExecutor(ops.asin);
    case 'Asinh':
      return unaryOpExecutor(ops.asinh);
    case 'Atan':
      return unaryOpExecutor(ops.atan);
    case 'Atan2': {
      const x$ = builder.param('x');
      const y$ = builder.param('y');
      return (ctx: ExecutionContext) => {
        return [ops.atan2(ctx.getOpParamValue(x$), ctx.getOpParamValue(y$))];
      }
    }
    case 'Atanh':
      return unaryOpExecutor(ops.atanh);
    case 'Ceil':
      return unaryOpExecutor(ops.ceil);
    case 'Complex': {
      const real$ = builder.param('real');
      const imag$ = builder.param('imag');
      return (ctx: ExecutionContext) => {
        return [ops.complex(
            ctx.getOpParamValue(real$), ctx.getOpParamValue(imag$))];
      }
    }
    case 'Cos':
      return unaryOpExecutor(ops.cos);
    case 'Cosh':
      return unaryOpExecutor(ops.cosh);
    case 'Elu':
      return unaryOpExecutor(ops.elu);
    case 'Erf':
      return unaryOpExecutor(ops.erf);
    case 'Exp':
      return unaryOpExecutor(ops.exp);
    case 'Expm1':
      return unaryOpExecutor(ops.expm1);
    case 'Floor':
      return unaryOpExecutor(ops.floor);
    case 'Log':
      return unaryOpExecutor(ops.log);
    case 'Log1p':
      return unaryOpExecutor(ops.log1p);
    case 'Imag':
      return unaryOpExecutor(ops.imag);

    case 'Neg':
      return unaryOpExecutor(ops.neg);
    case 'Reciprocal':
      return unaryOpExecutor(ops.reciprocal);
    case 'Real':
      return unaryOpExecutor(ops.real);
    case 'Relu':
      return unaryOpExecutor(ops.relu);
    case 'Round':
      return unaryOpExecutor(ops.round);
    case 'Selu':
      return unaryOpExecutor(ops.selu);
    case 'Sigmoid':
      return unaryOpExecutor(ops.sigmoid);
    case 'Sin':
      return unaryOpExecutor(ops.sin);
    case 'Sign':
      return unaryOpExecutor(ops.sign);
    case 'Sinh':
      return unaryOpExecutor(ops.sinh);
    case 'Softplus':
      return unaryOpExecutor(ops.softplus);
    case 'Sqrt':
      return unaryOpExecutor(ops.sqrt);
    case 'Square':
      return unaryOpExecutor(ops.square);
    case 'Tanh':
      return unaryOpExecutor(ops.tanh);
    case 'Tan':
      return unaryOpExecutor(ops.tan);
    case 'ClipByValue': {
      const x$ = builder.param('x');
      const clipValueMin$ = builder.param('clipValueMin');
      const clipValueMax$ = builder.param('clipValueMax');
      return (ctx: ExecutionContext) => {
        return [ops.clipByValue(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(clipValueMin$),
            ctx.getOpParamValue(clipValueMax$))];
      }
    }
    case 'Relu6':
      return unaryOpExecutor(ops.relu6);
    case 'Rsqrt':
      return unaryOpExecutor(ops.rsqrt);
    case 'Prod': {
      const x$ = builder.param('x');
      const axes$ = builder.param('axes');
      return (ctx: ExecutionContext) => {
        return [ops.prod(ctx.getOpParamValue(x$), ctx.getOpParamValue(axes$))];
      }
    }
    case 'LeakyRelu': {
      const x$ = builder.param('x');
      const alpha$ = builder.param('alpha');
      return (ctx: ExecutionContext) => {
        return [ops.leakyRelu(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(alpha$))];
      }
    }
    case 'Prelu': {
      const x$ = builder.param('x');
      const alpha$ = builder.param('alpha');
      return (ctx: ExecutionContext) => {
        return [ops.prelu(
            ctx.getOpParamValue(x$), ctx.getOpParamValue(alpha$))];
      }
    }
    case 'IsNan': {
      const info = builder.inputInfo(node.inputNames[0]);
      return (ctx: ExecutionContext) => {
        return [ops.isNaN(ctx.getOpParamValue(info))];
      }
    }
    case 'IsInf': {
      const info = builder.inputInfo(node.inputNames[0]);
      return (ctx: ExecutionContext) => {
        return [ops.isInf(ctx.getOpParamValue(info))];
      }
    }
    case 'IsFinite': {
      const info = builder.inputInfo(node.inputNames[0]);
      return (ctx: ExecutionContext) => {
        return [ops.isFinite(ctx.getOpParamValue(info))];
      }
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}

export const CATEGORY = 'basic_math';
