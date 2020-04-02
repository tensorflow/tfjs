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

import {MaxPoolWithArgmax, MaxPoolWithArgmaxAttrs, MaxPoolWithArgmaxInputs} from '../../../kernel_names';
import {KernelConfig} from '../../../kernel_registry';
import * as conv_util from '../../../ops/conv_util';
import * as util from '../../../util';
import {MathBackendWebGL} from '../backend_webgl';

import {maxPoolWithArgmaxImpl} from './MaxPoolWithArgmax_impl';

export const maxPoolWithArgmaxConfig: KernelConfig = {
  kernelName: MaxPoolWithArgmax,
  backendName: 'webgl',
  kernelFunc: ({inputs, attrs, backend}) => {
    const {x} = inputs as MaxPoolWithArgmaxInputs;
    const {filterSize, strides, pad, includeBatchInIndex} =
        attrs as {} as MaxPoolWithArgmaxAttrs;
    const webglBackend = backend as MathBackendWebGL;

    util.assert(
        x.shape.length === 4,
        () => `Error in maxPool: input must be rank 4 but got rank ${
            x.shape.length}.`);
    const dilations: [number, number] = [1, 1];
    util.assert(
        conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
        () => 'Error in maxPool: Either strides or dilations must be 1. ' +
            `Got strides ${strides} and dilations '${dilations}'`);

    const convInfo = conv_util.computePool2DInfo(
        x.shape as [number, number, number, number], filterSize, strides,
        dilations, pad);

    const [result, indexes] =
        maxPoolWithArgmaxImpl(x, includeBatchInIndex, convInfo, webglBackend);
    return [result, indexes];
  }
};
