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
  switch (node.op) {
    case 'BiasAdd':
    case 'AddV2':
    case 'Add': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.add(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'AddN': {
      const tensors$ = builder.param('tensors');
      return (ctx: ExecutionContext) => {
        return [ops.addN(ctx.getOpParamValue(tensors$))];
      };
    }
    case 'FloorMod':
    case 'Mod': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.mod(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'Mul': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.mul(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'RealDiv':
    case 'Div': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.div(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'DivNoNan': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.divNoNan(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'FloorDiv': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.floorDiv(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'Sub': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.sub(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'Minimum': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.minimum(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'Maximum': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.maximum(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'Pow': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.pow(ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    case 'SquaredDifference': {
      const a$ = builder.param('a');
      const b$ = builder.param('b');
      return (ctx: ExecutionContext) => {
        return [ops.squaredDifference(
            ctx.getOpParamValue(a$), ctx.getOpParamValue(b$))];
      }
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
}

export const CATEGORY = 'arithmetic';
