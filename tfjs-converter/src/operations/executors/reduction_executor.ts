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
import type {OpExecutorBuilder} from '../operation_executor';
import {InternalOpExecutor} from '../types';

export function buildOpExecutor(builder: OpExecutorBuilder):
    InternalOpExecutor {
  const node = builder.node;
  const ops = builder.manager.ops;

  function commonReductionOpExecutor(op: typeof ops.max): InternalOpExecutor {
    const x$ = builder.param('x');
    const axis$ = builder.param('axis');
    const keepDims$ = builder.param('keepDims');
    return (ctx: ExecutionContext) => [op(
               ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$),
               ctx.getOpParamValue(keepDims$))];
  }

  switch (node.op) {
    case 'Max':
      return commonReductionOpExecutor(ops.max);
    case 'Mean':
      return commonReductionOpExecutor(ops.mean);
    case 'Min':
      return commonReductionOpExecutor(ops.min);
    case 'Sum':
      return commonReductionOpExecutor(ops.sum);
    case 'All':
      return commonReductionOpExecutor(ops.all);
    case 'Any':
      return commonReductionOpExecutor(ops.any);
    case 'Prod':
      return commonReductionOpExecutor(ops.prod);
    case 'ArgMax': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      return (ctx: ExecutionContext) => [ops.argMax(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$))];
    }
    case 'ArgMin': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      return (ctx: ExecutionContext) => [ops.argMin(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$))];
    }
    case 'Cumprod': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      const exclusive$ = builder.param('exclusive');
      const reverse$ = builder.param('reverse');
      return (ctx: ExecutionContext) => [ops.cumprod(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$),
                 ctx.getOpParamValue(exclusive$),
                 ctx.getOpParamValue(reverse$))];
    }
    case 'Cumsum': {
      const x$ = builder.param('x');
      const axis$ = builder.param('axis');
      const exclusive$ = builder.param('exclusive');
      const reverse$ = builder.param('reverse');
      return (ctx: ExecutionContext) => [ops.cumsum(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(axis$),
                 ctx.getOpParamValue(exclusive$),
                 ctx.getOpParamValue(reverse$))];
    }
    case 'Bincount': {
      const x$ = builder.param('x');
      const weights$ = builder.param('weights');
      const size$ = builder.param('size');
      return (ctx: ExecutionContext) => [ops.bincount(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(weights$),
                 ctx.getOpParamValue(size$))];
    }
    case 'DenseBincount': {
      const x$ = builder.param('x');
      const weights$ = builder.param('weights');
      const size$ = builder.param('size');
      const binaryOutput$ = builder.param('binaryOutput');
      return (ctx: ExecutionContext) => [ops.denseBincount(
                 ctx.getOpParamValue(x$), ctx.getOpParamValue(weights$),
                 ctx.getOpParamValue(size$),
                 ctx.getOpParamValue(binaryOutput$))];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'reduction';
