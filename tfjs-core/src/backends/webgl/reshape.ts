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
import {webgl_util} from '../../webgl';

import {MathBackendWebGL} from './backend_webgl';
import {ReshapePackedProgram} from './reshape_packed_gpu';

const packedReshape =
    (input: TensorInfo, afterShape: number[],
     backend: MathBackendWebGL): TensorInfo => {
      const input3DShape = [
        webgl_util.getBatchDim(input.shape),
        ...webgl_util.getRowsCols(input.shape)
      ] as [number, number, number];
      const input3D: TensorInfo = {
        dtype: input.dtype,
        shape: input3DShape,
        dataId: input.dataId
      };
      const afterShapeAs3D = [
        webgl_util.getBatchDim(afterShape),
        ...webgl_util.getRowsCols(afterShape)
      ] as [number, number, number];

      const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
      const preventEagerUnpackingOfOutput = true;
      const output = backend.runWebGLProgram(
          program, [input3D], input.dtype, null /* customSetup */,
          preventEagerUnpackingOfOutput);
      return {dataId: output.dataId, shape: afterShape, dtype: output.dtype};
    };

export const reshape =
    (x: TensorInfo, afterShape: number[],
     backend: MathBackendWebGL): TensorInfo => {
      const xTexData = backend.texData.get(x.dataId);
      if (xTexData.isPacked && !webgl_util.isReshapeFree(x.shape, afterShape) &&
          !(xTexData.texture !== null &&
            webgl_util.isReshapeFree(xTexData.shape, afterShape))) {
        return packedReshape(x, afterShape, backend);
      }
      x.shape = afterShape;
      return x;
    };
