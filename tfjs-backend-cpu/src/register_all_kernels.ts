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
// We explicitly import the modular kernels so they get registered in the
// global registry when we compile the library. A modular build would replace
// the contents of this file and import only the kernels that are needed.
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {divConfig} from './kernels/Div';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {maxConfig} from './kernels/Max';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {transposeConfig} from './kernels/Transpose';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  dilation2dConfig, dilation2dBackpropInputConfig,
  dilation2dBackpropFilterConfig, divConfig, flipLeftRightConfig,
  maxPoolWithArgmaxConfig, maxConfig, nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config, rotateWithOffsetConfig, squareConfig,
  squaredDifferenceConfig, transposeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
