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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '../../../kernel_registry';
import * as axis_util from '../../../ops/axis_util';
import {computeOptimalWindowSize} from '../../../ops/reduce_util';
import {sizeFromShape} from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';
import {ReduceProgram} from '../reduce_gpu';

interface MaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface MaxAttrs extends NamedAttrMap {
  axes: number[];
}

export const maxImpl =
    (x: TensorInfo, backend: MathBackendWebGL): TensorInfo => {
      const [batchSize, inSize] = x.shape;
      const windowSize = computeOptimalWindowSize(inSize);
      const reduceInfo = {windowSize, inSize, batchSize};
      const program = new ReduceProgram(reduceInfo, 'max');
      const output = backend.runWebGLProgram(program, [x], x.dtype);

      if (output.shape[1] === 1) {
        return output;
      }
      return maxImpl(output, backend);
    };

registerKernel({
  kernelName: 'Max',
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxInputs;
    const {axes} = attrs as MaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    axis_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);
    const inSize = sizeFromShape(reduceShape);
    const xSize = sizeFromShape(x.shape);

    // TODO: Call reshape kernel.
    x.shape = [xSize / inSize, inSize];

    const out = maxImpl(x, webglBackend);

    return {dataId: out.dataId, shape: outShape, dtype: x.dtype};
  }
});
