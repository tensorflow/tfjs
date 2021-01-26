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

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {absConfig} from './kernels/Abs';
import {addConfig} from './kernels/Add';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {expandDimsConfig} from './kernels/ExpandDims';
import {expConfig} from './kernels/Exp';
import {fillConfig} from './kernels/Fill';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsAsyncConfig} from './kernels/FromPixelsAsync';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {minConfig} from './kernels/Min';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {negConfig} from './kernels/Neg';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {padV2Config} from './kernels/PadV2';
import {preluConfig} from './kernels/Prelu';
import {realDivConfig} from './kernels/RealDiv';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {selectConfig} from './kernels/Select';
import {sigmoidConfig} from './kernels/Sigmoid';
import {sliceConfig} from './kernels/Slice';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanhConfig} from './kernels/Tanh';
import {transposeConfig} from './kernels/Transpose';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  absConfig,
  addConfig,
  argMaxConfig,
  argMinConfig,
  avgPoolConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  castConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  cropAndResizeConfig,
  depthwiseConv2dNativeConfig,
  expandDimsConfig,
  expConfig,
  fillConfig,
  fromPixelsAsyncConfig,
  fromPixelsConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  maxConfig,
  maximumConfig,
  maxPoolConfig,
  minConfig,
  mirrorPadConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  padV2Config,
  preluConfig,
  realDivConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  selectConfig,
  sigmoidConfig,
  sliceConfig,
  stridedSliceConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  sumConfig,
  tanhConfig,
  transposeConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
