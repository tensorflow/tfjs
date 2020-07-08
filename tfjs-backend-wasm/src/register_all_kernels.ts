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

import {fusedMatMulConfig} from './kernels/_FusedMatMul';
import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {castConfig} from './kernels/Cast';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {cosConfig} from './kernels/Cos';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {depthwiseConv2DNativeConfig} from './kernels/DepthwiseConv2dNative';
import {divConfig} from './kernels/Div';
import {equalConfig} from './kernels/Equal';
import {expConfig} from './kernels/Exp';
import {fillConfig} from './kernels/Fill';
import {floorDivConfig} from './kernels/FloorDiv';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {gatherV2Config} from './kernels/GatherV2';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {minimumConfig} from './kernels/Minimum';
import {multiplyConfig} from './kernels/Multiply';
import {reverseConfig} from './kernels/Reverse';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  addConfig,
  addNConfig,
  argMaxConfig,
  avgPoolConfig,
  batchMatMulConfig,
  castConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropInputConfig,
  cosConfig,
  cropAndResizeConfig,
  depthwiseConv2DNativeConfig,
  divConfig,
  equalConfig,
  expConfig,
  fillConfig,
  floorDivConfig,
  fusedMatMulConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  gatherV2Config,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  logicalAndConfig,
  maxConfig,
  maximumConfig,
  minimumConfig,
  multiplyConfig,
  reverseConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
