/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {addGradConfig} from './gradients/Add_grad';
import {addNGradConfig} from './gradients/AddN_grad';
import {atan2GradConfig} from './gradients/Atan2_grad';
import {avgPool3DGradConfig} from './gradients/AvgPool3D_grad';
import {avgPoolGradConfig} from './gradients/AvgPool_grad';
import {batchMatMulGradConfig} from './gradients/BatchMatMul_grad';
import {batchToSpaceNDGradConfig} from './gradients/BatchToSpaceND_grad';
import {broadcastToGradConfig} from './gradients/BroadcastTo_grad';
import {concatGradConfig} from './gradients/Concat_grad';
import {conv2DGradConfig} from './gradients/Conv2D_grad';
import {conv2DBackpropInputGradConfig} from './gradients/Conv2DBackpropInput_grad';
import {conv3DGradConfig} from './gradients/Conv3D_grad';
import {cumsumGradConfig} from './gradients/Cumsum_grad';
import {depthwiseConv2dNativeGradConfig} from './gradients/DepthwiseConv2dNative_grad';
import {divGradConfig} from './gradients/Div_grad';
import {eluGradConfig} from './gradients/Elu_grad';
import {floorDivGradConfig} from './gradients/FloorDiv_grad';
import {fusedBatchNormGradConfig} from './gradients/FusedBatchNorm_grad';
import {greaterEqualGradConfig} from './gradients/GreaterEqual_grad';
import {identityGradConfig} from './gradients/Identity_grad';
import {lrnGradConfig} from './gradients/LRN_grad';
import {maxGradConfig} from './gradients/Max_grad';
import {maximumGradConfig} from './gradients/Maximum_grad';
import {maxPool3DGradConfig} from './gradients/MaxPool3D_grad';
import {maxPoolGradConfig} from './gradients/MaxPool_grad';
import {minimumGradConfig} from './gradients/Minimum_grad';
import {modGradConfig} from './gradients/Mod_grad';
import {multiplyGradConfig} from './gradients/Multiply_grad';
import {oneHotGradConfig} from './gradients/OneHot_grad';
import {padV2GradConfig} from './gradients/PadV2_grad';
import {powGradConfig} from './gradients/Pow_grad';
import {preluGradConfig} from './gradients/Prelu_grad';
import {relu6GradConfig} from './gradients/Relu6_grad';
import {reluGradConfig} from './gradients/Relu_grad';
import {seluGradConfig} from './gradients/Selu_grad';
import {spaceToBatchNDGradConfig} from './gradients/SpaceToBatchND_grad';
import {splitVGradConfig} from './gradients/SplitV_grad';
import {squareGradConfig} from './gradients/Square_grad';
import {squaredDifferenceGradConfig} from './gradients/SquaredDifference_grad';
import {subGradConfig} from './gradients/Sub_grad';
import {tileGradConfig} from './gradients/Tile_grad';
import {transposeGradConfig} from './gradients/Transpose_grad';
import {GradConfig} from './kernel_registry';
import {registerGradient} from './kernel_registry';

// Export all kernel configs here so that the package can auto register them
const gradConfigs: GradConfig[] = [
  addGradConfig,
  addNGradConfig,
  atan2GradConfig,
  avgPoolGradConfig,
  avgPool3DGradConfig,
  batchMatMulGradConfig,
  batchToSpaceNDGradConfig,
  broadcastToGradConfig,
  concatGradConfig,
  conv2DGradConfig,
  conv2DBackpropInputGradConfig,
  conv3DGradConfig,
  cumsumGradConfig,
  depthwiseConv2dNativeGradConfig,
  divGradConfig,
  eluGradConfig,
  floorDivGradConfig,
  fusedBatchNormGradConfig,
  greaterEqualGradConfig,
  identityGradConfig,
  lrnGradConfig,
  oneHotGradConfig,
  padV2GradConfig,
  splitVGradConfig,
  maxGradConfig,
  spaceToBatchNDGradConfig,
  maxGradConfig,
  maximumGradConfig,
  maxPoolGradConfig,
  maxPool3DGradConfig,
  minimumGradConfig,
  modGradConfig,
  multiplyGradConfig,
  oneHotGradConfig,
  padV2GradConfig,
  powGradConfig,
  preluGradConfig,
  reluGradConfig,
  relu6GradConfig,
  seluGradConfig,
  spaceToBatchNDGradConfig,
  splitVGradConfig,
  squareGradConfig,
  squaredDifferenceGradConfig,
  tileGradConfig,
  transposeGradConfig,
  subGradConfig
];

for (const gradientConfig of gradConfigs) {
  registerGradient(gradientConfig);
}
