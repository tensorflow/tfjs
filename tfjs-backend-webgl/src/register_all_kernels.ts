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
import {acosConfig} from './kernels/Acos';
import {acoshConfig} from './kernels/Acosh';
import {addConfig} from './kernels/Add';
import {allConfig} from './kernels/All';
import {anyConfig} from './kernels/Any';
import {asinConfig} from './kernels/Asin';
import {asinhConfig} from './kernels/Asinh';
import {atanConfig} from './kernels/Atan';
import {atan2Config} from './kernels/Atan2';
import {atanhConfig} from './kernels/Atanh';
import {avgPoolConfig} from './kernels/AvgPool';
import {avgPool3DConfig} from './kernels/AvgPool3D';
import {avgPoolGrad3DConfig} from './kernels/AvgPool3DGrad';
import {avgPoolGradConfig} from './kernels/AvgPoolGrad';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchNormConfig} from './kernels/BatchNorm';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropFilterConfig} from './kernels/Conv2DBackpropFilter';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {conv3DConfig} from './kernels/Conv3D';
import {conv3DBackpropFilterV2Config} from './kernels/Conv3DBackpropFilterV2';
import {conv3DBackpropInputConfig} from './kernels/Conv3DBackpropInputV2';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {cumsumConfig} from './kernels/Cumsum';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {depthwiseConv2dNativeBackpropFilterConfig} from './kernels/DepthwiseConv2dNativeBackpropFilter';
import {depthwiseConv2dNativeBackpropInputConfig} from './kernels/DepthwiseConv2dNativeBackpropInput';
import {diagConfig} from './kernels/Diag';
import {eluGradConfig} from './kernels/EluGrad';
import {equalConfig} from './kernels/Equal';
import {erfConfig} from './kernels/Erf';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
import {fftConfig} from './kernels/FFT';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {ifftConfig} from './kernels/IFFT';
import {imagConfig} from './kernels/Imag';
import {isFiniteConfig} from './kernels/IsFinite';
import {isInfConfig} from './kernels/IsInf';
import {isNaNConfig} from './kernels/IsNaN';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {linSpaceConfig} from './kernels/LinSpace';
import {log1pConfig} from './kernels/Log1p';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {logicalOrConfig} from './kernels/LogicalOr';
import {LRNConfig} from './kernels/LRN';
import {LRNGradConfig} from './kernels/LRNGrad';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {maxPool3DConfig} from './kernels/MaxPool3D';
import {maxPoolGrad3DConfig} from './kernels/MaxPool3DGrad';
import {maxPoolGradConfig} from './kernels/MaxPoolGrad';
import {maxPoolWithArgmaxConfig} from './kernels/MaxPoolWithArgmax';
import {meanConfig} from './kernels/Mean';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {modConfig} from './kernels/Mod';
import {multinomialConfig} from './kernels/Multinomial';
import {multiplyConfig} from './kernels/Multiply';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV4Config} from './kernels/NonMaxSuppressionV4';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {oneHotConfig} from './kernels/OneHot';
import {onesLikeConfig} from './kernels/OnesLike';
import {packConfig} from './kernels/Pack';
import {padV2Config} from './kernels/PadV2';
import {powConfig} from './kernels/Pow';
import {preluConfig} from './kernels/Prelu';
import {prodConfig} from './kernels/Prod';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reciprocalConfig} from './kernels/Reciprocal';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {resizeBilinearGradConfig} from './kernels/ResizeBilinearGrad';
import {resizeNearestNeighborConfig} from './kernels/ResizeNearestNeighbor';
import {resizeNearestNeighborGradConfig} from './kernels/ResizeNearestNeighborGrad';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {roundConfig} from './kernels/Round';
import {scatterNdConfig} from './kernels/ScatterNd';
import {seluConfig} from './kernels/Selu';
import {sigmoidConfig} from './kernels/Sigmoid';
import {signConfig} from './kernels/Sign';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {softplusConfig} from './kernels/Softplus';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseToDenseConfig} from './kernels/SparseToDense';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stepConfig} from './kernels/Step';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
import {tanhConfig} from './kernels/Tanh';
import {tileConfig} from './kernels/Tile';
import {transposeConfig} from './kernels/Transpose';
import {uniqueConfig} from './kernels/Unique';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  LRNConfig,
  LRNGradConfig,
  _fusedMatMulConfig,
  acosConfig,
  acoshConfig,
  addConfig,
  allConfig,
  anyConfig,
  asinConfig,
  asinhConfig,
  atan2Config,
  atanConfig,
  atanhConfig,
  avgPool3DConfig,
  avgPoolConfig,
  avgPoolGrad3DConfig,
  avgPoolGradConfig,
  batchMatMulConfig,
  batchNormConfig,
  batchToSpaceNDConfig,
  castConfig,
  complexConfig,
  concatConfig,
  conv2DBackpropFilterConfig,
  conv2DBackpropInputConfig,
  conv2DConfig,
  conv3DBackpropFilterV2Config,
  conv3DBackpropInputConfig,
  conv3DConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumsumConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeBackpropFilterConfig,
  depthwiseConv2dNativeBackpropFilterConfig,
  depthwiseConv2dNativeBackpropInputConfig,
  depthwiseConv2dNativeBackpropInputConfig,
  depthwiseConv2dNativeConfig,
  depthwiseConv2dNativeConfig,
  diagConfig,
  eluGradConfig,
  equalConfig,
  erfConfig,
  expConfig,
  expandDimsConfig,
  fftConfig,
  fillConfig,
  flipLeftRightConfig,
  floorConfig,
  floorDivConfig,
  fromPixelsConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  ifftConfig,
  imagConfig,
  isFiniteConfig,
  isInfConfig,
  isNaNConfig,
  lessConfig,
  lessEqualConfig,
  linSpaceConfig,
  log1pConfig,
  logicalAndConfig,
  logicalNotConfig,
  logicalOrConfig,
  maxConfig,
  maxPool3DConfig,
  maxPoolConfig,
  maxPoolGrad3DConfig,
  maxPoolGradConfig,
  maxPoolWithArgmaxConfig,
  maximumConfig,
  meanConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  modConfig,
  multinomialConfig,
  multiplyConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV4Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  oneHotConfig,
  onesLikeConfig,
  packConfig,
  padV2Config,
  powConfig,
  preluConfig,
  prodConfig,
  realConfig,
  realDivConfig,
  reciprocalConfig,
  reshapeConfig,
  resizeBilinearConfig,
  resizeBilinearGradConfig,
  resizeNearestNeighborConfig,
  resizeNearestNeighborGradConfig,
  rotateWithOffsetConfig,
  roundConfig,
  scatterNdConfig,
  seluConfig,
  sigmoidConfig,
  signConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  softmaxConfig,
  softplusConfig,
  spaceToBatchNDConfig,
  sparseToDenseConfig,
  squareConfig,
  squaredDifferenceConfig,
  stepConfig,
  subConfig,
  sumConfig,
  tanConfig,
  tanhConfig,
  tileConfig,
  transposeConfig,
  uniqueConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
