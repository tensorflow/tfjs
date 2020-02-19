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
import {sumOutType} from '../../../types';
import {sizeFromShape} from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';
import {ReduceProgram} from '../reduce_gpu';
import {reshape} from '../reshape';

interface SumInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SumAttrs extends NamedAttrMap {
  axes: number[];
}

export const sumImpl =
    (x: TensorInfo, reduceShape: number[], outShape: number[],
     backend: MathBackendWebGL): TensorInfo => {
      const inSize = sizeFromShape(reduceShape);
      const xSize = sizeFromShape(x.shape);
      const batchSize = xSize / inSize;

      x = reshape(x, [batchSize, inSize], backend);

      const windowSize = computeOptimalWindowSize(inSize);
      const reduceInfo = {windowSize, inSize, batchSize};
      const program = new ReduceProgram(reduceInfo, 'sum');
      const output = backend.runWebGLProgram(program, [x], sumOutType(x.dtype));

      if (output.shape[1] === 1) {
        return reshape(output, outShape, backend);
      }
      return sumImpl(output, reduceShape, outShape, backend);
    };

registerKernel({
  kernelName: 'Sum',
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as SumInputs;
    const {axes} = attrs as SumAttrs;
    const webglBackend = backend as MathBackendWebGL;

    axis_util.assertAxesAreInnerMostDims('sum', axes, x.shape.length);

    const [outShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(x.shape, axes);

    const out = sumImpl(x, reduceShape, outShape, webglBackend);

    return {dataId: out.dataId, shape: outShape, dtype: x.dtype};
  }
});
