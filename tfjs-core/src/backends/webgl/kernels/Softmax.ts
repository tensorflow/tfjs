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
import {parseAxisParam, sizeFromShape} from '../../../util';
import {webgl_util} from '../../../webgl';
import {MathBackendWebGL} from '../backend_webgl';

interface SoftmaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SoftmaxAttrs extends NamedAttrMap {
  dim: number;
}

registerKernel({
  kernelName: 'Softmax',
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {logits} = inputs as SoftmaxInputs;
    const {dim} = attrs as SoftmaxAttrs;
    const webglBackend = backend as MathBackendWebGL;
    // const textureManager = webglBackend.getTextureManager();

    const axes = parseAxisParam([dim], logits.shape);

    const [reduceOutShape, reduceShape] =
        axis_util.computeOutAndReduceShapes(logits.shape, axes);
    // const logitsValues =
    //     webglBackend.data.get(logits.dataId).values as Float32Array;
    // const maxLogit =
    //     max(logitsValues, reduceShape,
    //         new Float32Array(sizeFromShape(reduceOutShape)));


    // Create resource for max kernel:
    const logitsTexdata = webglBackend.texData.get(logits.dataId);

    const texShapeForMax =
        webgl_util.getTextureShapeFromLogicalShape(reduceOutShape);

    // Not sure about usage - check webglBackend.runWebglProgram for reference
    // on shader program output usage.
    const texUsage: any = null;
    // This is only the texture... need to create texData object.
    const texForMax = webglBackend.acquireTexture(
        texShapeForMax, texUsage, logits.dtype, logitsTexdata.isPacked);

    // const maxLogit = max(
    //     logitsTexdata,
    //     reduceShape,
    // );

    console.log('MAX');
    console.log(texForMax, reduceShape, sizeFromShape);

    return {dataId: logits.dataId, shape: logits.shape, dtype: logits.dtype};

    // const dataId = webglBackend.write(maxLogit, logits.shape, logits.dtype);
    // return {dataId, shape: logits.shape, dtype: logits.dtype};
  }
});
