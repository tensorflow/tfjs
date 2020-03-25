/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {TensorInfo} from '../../kernel_registry';
import {computeOptimalWindowSize} from '../../ops/reduce_util';
import {DataType} from '../../types';

import {MathBackendWebGL} from './backend_webgl';
import {ReduceProgram} from './reduce_gpu';

export const reduce =
    (x: TensorInfo, reduceShape: number[], dtype: DataType, outInfo: TensorInfo,
     backend: MathBackendWebGL): TensorInfo => {
      const [batchSize, inSize] = x.shape;
      const windowSize = computeOptimalWindowSize(inSize);
      const reduceInfo = {windowSize, inSize, batchSize};
      const program = new ReduceProgram(reduceInfo, 'sum');
      const output = backend.runWebGLProgram(program, [x], dtype, outInfo);

      backend.disposeData(x.dataId);

      if (output.shape[1] === 1) {
        return output;
      }

      const [newBatchSize, newInSize] = output.shape;
      const newWindowSize = computeOptimalWindowSize(newInSize);
      const newOutShape = [newBatchSize, Math.ceil(newInSize / newWindowSize)];

      const newOutInfo = backend.makeTensorInfo(newOutShape, output.dtype);

      return reduce(output, reduceShape, dtype, newOutInfo, backend);
    };
