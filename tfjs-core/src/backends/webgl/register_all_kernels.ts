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
import {KernelConfig, registerKernel} from '../../kernel_registry';

import {fromPixelsConfig} from './kernels/FromPixels';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  fromPixelsConfig,
  nonMaxSuppressionV5Config,
  squareConfig,
  squaredDifferenceConfig,
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
