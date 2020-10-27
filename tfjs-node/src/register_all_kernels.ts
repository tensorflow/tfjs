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

import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {diagConfig} from './kernels/Diag';
import {dilation2dConfig} from './kernels/Dilation2D';
import {dilation2dBackpropFilterConfig} from './kernels/Dilation2DBackpropFilter';
import {dilation2dBackpropInputConfig} from './kernels/Dilation2DBackpropInput';
import {equalConfig} from './kernels/Equal';
import {floorDivConfig} from './kernels/FloorDiv';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {prodConfig} from './kernels/Prod';
import {softmaxConfig} from './kernels/Softmax';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {sumConfig} from './kernels/Sum';
import {unsortedSegmentSumConfig} from './kernels/UnsortedSegmentSum';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  addConfig,
  addNConfig,
  argMaxConfig,
  argMinConfig,
  diagConfig,
  dilation2dBackpropFilterConfig,
  dilation2dBackpropInputConfig,
  dilation2dConfig,
  equalConfig,
  floorDivConfig,
  greaterConfig,
  greaterEqualConfig,
  lessConfig,
  lessEqualConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  maxConfig,
  maximumConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  multiplyConfig,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  prodConfig,
  softmaxConfig,
  squaredDifferenceConfig,
  sumConfig,
  unsortedSegmentSumConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
