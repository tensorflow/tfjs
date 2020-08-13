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
import {KernelConfig, registerKernel} from '@tensorflow/tfjs-core';

import {divConfig} from './kernels/Div';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {fromPixelsConfig} from './kernels/FromPixels';
import {maxConfig} from './kernels/Max';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {transposeConfig} from './kernels/Transpose';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  maxConfig, flipLeftRightConfig, fromPixelsConfig, divConfig,
  maxPoolWithArgmaxConfig, nonMaxSuppressionV3Config, nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config, rotateWithOffsetConfig, squareConfig,
  squaredDifferenceConfig, transposeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
