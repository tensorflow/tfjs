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
import {absGradConfig} from './gradients/Abs_grad';
import {acosGradConfig} from './gradients/Acos_grad';
import {acoshGradConfig} from './gradients/Acosh_grad';
import {addGradConfig} from './gradients/Add_grad';
import {addNGradConfig} from './gradients/AddN_grad';
import {argMaxGradConfig} from './gradients/ArgMax_grad';
import {argMinGradConfig} from './gradients/ArgMin_grad';
import {asinGradConfig} from './gradients/Asin_grad';
import {asinhGradConfig} from './gradients/Asinh_grad';
import {atan2GradConfig} from './gradients/Atan2_grad';
import {atanGradConfig} from './gradients/Atan_grad';
import {atanhGradConfig} from './gradients/Atanh_grad';
import {avgPool3DGradConfig} from './gradients/AvgPool3D_grad';
import {avgPoolGradConfig} from './gradients/AvgPool_grad';
import {batchMatMulGradConfig} from './gradients/BatchMatMul_grad';
import {batchToSpaceNDGradConfig} from './gradients/BatchToSpaceND_grad';
import {broadcastToGradConfig} from './gradients/BroadcastTo_grad';
import {castGradConfig} from './gradients/Cast_grad';
import {ceilGradConfig} from './gradients/Ceil_grad';
import {clipByValueGradConfig} from './gradients/ClipByValue_grad';
import {complexAbsGradConfig} from './gradients/ComplexAbs_grad';
import {concatGradConfig} from './gradients/Concat_grad';
import {conv2DGradConfig} from './gradients/Conv2D_grad';
import {conv2DBackpropInputGradConfig} from './gradients/Conv2DBackpropInput_grad';
import {conv3DGradConfig} from './gradients/Conv3D_grad';
import {cosGradConfig} from './gradients/Cos_grad';
import {coshGradConfig} from './gradients/Cosh_grad';
import {cumsumGradConfig} from './gradients/Cumsum_grad';
import {depthwiseConv2dNativeGradConfig} from './gradients/DepthwiseConv2dNative_grad';
import {dilation2dGradConfig} from './gradients/Dilation2D_grad';
import {eluGradConfig} from './gradients/Elu_grad';
import {erfGradConfig} from './gradients/Erf_grad';
import {expGradConfig} from './gradients/Exp_grad';
import {expandDimsGradConfig} from './gradients/ExpandDims_grad';
import {expm1GradConfig} from './gradients/Expm1_grad';
import {floorGradConfig} from './gradients/Floor_grad';
import {floorDivGradConfig} from './gradients/FloorDiv_grad';
import {fusedBatchNormGradConfig} from './gradients/FusedBatchNorm_grad';
import {gatherGradConfig} from './gradients/GatherV2_grad';
import {greaterEqualGradConfig} from './gradients/GreaterEqual_grad';
import {identityGradConfig} from './gradients/Identity_grad';
import {isFiniteGradConfig} from './gradients/IsFinite_grad';
import {isInfGradConfig} from './gradients/IsInf_grad';
import {isNanGradConfig} from './gradients/IsNan_grad';
import {leakyReluGradConfig} from './gradients/LeakyRelu_grad';
import {log1pGradConfig} from './gradients/Log1p_grad';
import {logGradConfig} from './gradients/Log_grad';
import {logSoftmaxGradConfig} from './gradients/LogSoftmax_grad';
import {lrnGradConfig} from './gradients/LRN_grad';
import {maxGradConfig} from './gradients/Max_grad';
import {maximumGradConfig} from './gradients/Maximum_grad';
import {maxPool3DGradConfig} from './gradients/MaxPool3D_grad';
import {maxPoolGradConfig} from './gradients/MaxPool_grad';
import {meanGradConfig} from './gradients/Mean_grad';
import {minGradConfig} from './gradients/Min_grad';
import {minimumGradConfig} from './gradients/Minimum_grad';
import {mirrorPadGradConfig} from './gradients/MirrorPad_grad';
import {modGradConfig} from './gradients/Mod_grad';
import {multiplyGradConfig} from './gradients/Multiply_grad';
import {negGradConfig} from './gradients/Neg_grad';
import {oneHotGradConfig} from './gradients/OneHot_grad';
import {onesLikeGradConfig} from './gradients/OnesLike_grad';
import {packGradConfig} from './gradients/Pack_grad';
import {padV2GradConfig} from './gradients/PadV2_grad';
import {powGradConfig} from './gradients/Pow_grad';
import {preluGradConfig} from './gradients/Prelu_grad';
import {divGradConfig} from './gradients/RealDiv_grad';
import {reciprocalGradConfig} from './gradients/Reciprocal_grad';
import {relu6GradConfig} from './gradients/Relu6_grad';
import {reluGradConfig} from './gradients/Relu_grad';
import {reshapeGradConfig} from './gradients/Reshape_grad';
import {resizeBilinearGradConfig} from './gradients/ResizeBilinear_grad';
import {resizeNearestNeighborGradConfig} from './gradients/ResizeNearestNeighbor_grad';
import {reverseGradConfig} from './gradients/Reverse_grad';
import {roundGradConfig} from './gradients/Round_grad';
import {rsqrtGradConfig} from './gradients/Rsqrt_grad';
import {selectGradConfig} from './gradients/Select_grad';
import {seluGradConfig} from './gradients/Selu_grad';
import {sigmoidGradConfig} from './gradients/Sigmoid_grad';
import {signGradConfig} from './gradients/Sign_grad';
import {sinGradConfig} from './gradients/Sin_grad';
import {sinhGradConfig} from './gradients/Sinh_grad';
import {sliceGradConfig} from './gradients/Slice_grad';
import {softmaxGradConfig} from './gradients/Softmax_grad';
import {softplusGradConfig} from './gradients/Softplus_grad';
import {spaceToBatchNDGradConfig} from './gradients/SpaceToBatchND_grad';
import {splitVGradConfig} from './gradients/SplitV_grad';
import {sqrtGradConfig} from './gradients/Sqrt_grad';
import {squareGradConfig} from './gradients/Square_grad';
import {squaredDifferenceGradConfig} from './gradients/SquaredDifference_grad';
import {stepGradConfig} from './gradients/Step_grad';
import {subGradConfig} from './gradients/Sub_grad';
import {sumGradConfig} from './gradients/Sum_grad';
import {tanGradConfig} from './gradients/Tan_grad';
import {tanhGradConfig} from './gradients/Tanh_grad';
import {tileGradConfig} from './gradients/Tile_grad';
import {transposeGradConfig} from './gradients/Transpose_grad';
import {unpackGradConfig} from './gradients/Unpack_grad';
import {unsortedSegmentSumGradConfig} from './gradients/UnsortedSegmentSum_grad';
import {zerosLikeGradConfig} from './gradients/ZerosLike_grad';
import {GradConfig} from './kernel_registry';
import {registerGradient} from './kernel_registry';

// Export all kernel configs here so that the package can auto register them
const gradConfigs: GradConfig[] = [
  absGradConfig,
  acosGradConfig,
  acoshGradConfig,
  addGradConfig,
  addNGradConfig,
  argMaxGradConfig,
  argMinGradConfig,
  asinGradConfig,
  asinhGradConfig,
  atan2GradConfig,
  atanGradConfig,
  atanhGradConfig,
  avgPool3DGradConfig,
  avgPoolGradConfig,
  batchMatMulGradConfig,
  batchToSpaceNDGradConfig,
  broadcastToGradConfig,
  castGradConfig,
  ceilGradConfig,
  clipByValueGradConfig,
  complexAbsGradConfig,
  concatGradConfig,
  conv2DBackpropInputGradConfig,
  conv2DGradConfig,
  conv3DGradConfig,
  cosGradConfig,
  coshGradConfig,
  cumsumGradConfig,
  depthwiseConv2dNativeGradConfig,
  dilation2dGradConfig,
  divGradConfig,
  eluGradConfig,
  erfGradConfig,
  expGradConfig,
  expandDimsGradConfig,
  expm1GradConfig,
  floorDivGradConfig,
  floorGradConfig,
  fusedBatchNormGradConfig,
  gatherGradConfig,
  greaterEqualGradConfig,
  identityGradConfig,
  isFiniteGradConfig,
  isInfGradConfig,
  isNanGradConfig,
  leakyReluGradConfig,
  log1pGradConfig,
  logGradConfig,
  logSoftmaxGradConfig,
  lrnGradConfig,
  maxGradConfig,
  maxGradConfig,
  maximumGradConfig,
  maxPool3DGradConfig,
  maxPoolGradConfig,
  meanGradConfig,
  minGradConfig,
  minimumGradConfig,
  mirrorPadGradConfig,
  modGradConfig,
  multiplyGradConfig,
  negGradConfig,
  oneHotGradConfig,
  onesLikeGradConfig,
  packGradConfig,
  padV2GradConfig,
  padV2GradConfig,
  powGradConfig,
  preluGradConfig,
  reciprocalGradConfig,
  relu6GradConfig,
  reluGradConfig,
  reshapeGradConfig,
  resizeBilinearGradConfig,
  resizeNearestNeighborGradConfig,
  reverseGradConfig,
  roundGradConfig,
  rsqrtGradConfig,
  selectGradConfig,
  seluGradConfig,
  sigmoidGradConfig,
  signGradConfig,
  sinGradConfig,
  sinhGradConfig,
  sliceGradConfig,
  softmaxGradConfig,
  softplusGradConfig,
  spaceToBatchNDGradConfig,
  spaceToBatchNDGradConfig,
  splitVGradConfig,
  splitVGradConfig,
  sqrtGradConfig,
  squaredDifferenceGradConfig,
  squareGradConfig,
  stepGradConfig,
  subGradConfig,
  sumGradConfig,
  tanGradConfig,
  tanhGradConfig,
  tileGradConfig,
  transposeGradConfig,
  unpackGradConfig,
  unsortedSegmentSumGradConfig,
  zerosLikeGradConfig
];

for (const gradientConfig of gradConfigs) {
  registerGradient(gradientConfig);
}
