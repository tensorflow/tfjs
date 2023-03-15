/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {PoolWithFilterSizeEqualsOneProgram} from '../pool_filtersizeone_webgpu';
import {Pool2DProgram} from '../pool_webgpu';

import {identity} from './Identity';
import {max} from './Max';
import {mean} from './Mean';
import {reshape} from './Reshape';

type PoolType = 'max'|'avg';
export function poolImpl(
    x: TensorInfo, convInfo: backend_util.Conv2DInfo, poolType: PoolType,
    backend: WebGPUBackend): TensorInfo {
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return identity({inputs: {x}, backend});
  }

  if (convInfo.filterWidth === convInfo.inWidth &&
      convInfo.filterHeight === convInfo.inHeight && convInfo.batchSize === 1 &&
      convInfo.padInfo.type === 'VALID') {
    const length = x.shape.length;
    const reshapeX = reshape({
      inputs: {x},
      backend,
      attrs: {
        shape: [
          x.shape[length - 3] * x.shape[length - 2] /* height * width */,
          x.shape[length - 1] /* channel */
        ]
      }
    });
    let reduceX;
    if (poolType === 'avg') {
      reduceX = mean(
          {inputs: {x: reshapeX}, backend, attrs: {axis: 0, keepDims: false}});
    } else {
      util.assert(poolType === 'max', () => `Invalid pool type ${poolType}`);
      reduceX = max({
        inputs: {x: reshapeX},
        backend,
        attrs: {reductionIndices: 0, keepDims: false}
      });
    }

    const result = reshape(
        {inputs: {x: reduceX}, backend, attrs: {shape: convInfo.outShape}});
    backend.disposeData(reshapeX.dataId);
    backend.disposeData(reduceX.dataId);
    return result;
  }

  let program: Pool2DProgram|PoolWithFilterSizeEqualsOneProgram;
  const dimensions =
      [{type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]}];
  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
    program = new PoolWithFilterSizeEqualsOneProgram(convInfo);
  } else {
    if (poolType === 'avg') {
      program = new Pool2DProgram(convInfo, 'avg');
    } else {
      util.assert(poolType === 'max', () => `Invalid pool type ${poolType}`);
      program = new Pool2DProgram(convInfo, 'max');
    }

    dimensions.push(
        {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]}, {
          type: 'int32',
          data: [convInfo.dilationHeight, convInfo.dilationWidth]
        },
        {type: 'int32', data: [convInfo.inHeight, convInfo.inWidth]}, {
          type: 'int32',
          data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        });
  }

  return backend.runWebGPUProgram(program, [x], x.dtype, dimensions);
}
