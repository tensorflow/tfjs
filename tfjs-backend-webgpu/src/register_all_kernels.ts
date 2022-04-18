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
import {addNConfig} from './kernels/AddN';
import {argMaxConfig} from './kernels/ArgMax';
import {argMinConfig} from './kernels/ArgMin';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipByValueConfig} from './kernels/ClipByValue';
import {complexConfig} from './kernels/Complex';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {cumprodConfig} from './kernels/Cumprod';
import {cumsumConfig} from './kernels/Cumsum';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {einsumConfig} from './kernels/Einsum';
import {eluConfig} from './kernels/Elu';
import {equalConfig} from './kernels/Equal';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
import {expm1Config} from './kernels/Expm1';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fromPixelsConfig} from './kernels/FromPixels';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {gatherV2Config} from './kernels/GatherV2';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {imagConfig} from './kernels/Imag';
import {leakyReluConfig} from './kernels/LeakyRelu';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {logicalAndConfig} from './kernels/LogicalAnd';
import {logicalNotConfig} from './kernels/LogicalNot';
import {maxConfig} from './kernels/Max';
import {maximumConfig} from './kernels/Maximum';
import {maxPoolConfig} from './kernels/MaxPool';
import {meanConfig} from './kernels/Mean';
import {minConfig} from './kernels/Min';
import {minimumConfig} from './kernels/Minimum';
import {mirrorPadConfig} from './kernels/MirrorPad';
import {multiplyConfig} from './kernels/Multiply';
import {negConfig} from './kernels/Neg';
import {nonMaxSuppressionV3Config} from './kernels/NonMaxSuppressionV3';
import {nonMaxSuppressionV5Config} from './kernels/NonMaxSuppressionV5';
import {notEqualConfig} from './kernels/NotEqual';
import {onesLikeConfig} from './kernels/OnesLike';
import {packConfig} from './kernels/Pack';
import {padV2Config} from './kernels/PadV2';
import {powConfig} from './kernels/Pow';
import {preluConfig} from './kernels/Prelu';
import {prodConfig} from './kernels/Prod';
import {rangeConfig} from './kernels/Range';
import {realConfig} from './kernels/Real';
import {realDivConfig} from './kernels/RealDiv';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {resizeNearestNeighborConfig} from './kernels/ResizeNearestNeighbor';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {rsqrtConfig} from './kernels/Rsqrt';
import {scatterNdConfig} from './kernels/ScatterNd';
import {selectConfig} from './kernels/Select';
import {sigmoidConfig} from './kernels/Sigmoid';
import {sinConfig} from './kernels/Sin';
import {sinhConfig} from './kernels/Sinh';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseToDenseConfig} from './kernels/SparseToDense';
import {splitVConfig} from './kernels/SplitV';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {stringNGramsConfig} from './kernels/StringNGrams';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanhConfig} from './kernels/Tanh';
import {tileConfig} from './kernels/Tile';
import {topKConfig} from './kernels/TopK';
import {transformConfig} from './kernels/Transform';
import {transposeConfig} from './kernels/Transpose';
import {unpackConfig} from './kernels/Unpack';
import {zerosLikeConfig} from './kernels/ZerosLike';

// List all kernel configs here
const kernelConfigs: KernelConfig[] = [
  _fusedMatMulConfig,
  absConfig,
  addConfig,
  addNConfig,
  argMaxConfig,
  argMinConfig,
  avgPoolConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  complexConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropInputConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumprodConfig,
  cumsumConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeConfig,
  einsumConfig,
  eluConfig,
  equalConfig,
  expConfig,
  expandDimsConfig,
  expm1Config,
  fillConfig,
  flipLeftRightConfig,
  fromPixelsConfig,
  floorConfig,
  floorDivConfig,
  fusedBatchNormConfig,
  fusedConv2DConfig,
  fusedDepthwiseConv2DConfig,
  gatherNdConfig,
  gatherV2Config,
  greaterConfig,
  greaterEqualConfig,
  identityConfig,
  imagConfig,
  leakyReluConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  logicalAndConfig,
  logicalNotConfig,
  maxConfig,
  maximumConfig,
  maxPoolConfig,
  meanConfig,
  minConfig,
  minimumConfig,
  mirrorPadConfig,
  multiplyConfig,
  negConfig,
  nonMaxSuppressionV3Config,
  nonMaxSuppressionV5Config,
  notEqualConfig,
  onesLikeConfig,
  packConfig,
  padV2Config,
  powConfig,
  preluConfig,
  prodConfig,
  rangeConfig,
  realConfig,
  realDivConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  resizeNearestNeighborConfig,
  rotateWithOffsetConfig,
  rsqrtConfig,
  scatterNdConfig,
  selectConfig,
  sigmoidConfig,
  sinConfig,
  sinhConfig,
  sliceConfig,
  stridedSliceConfig,
  stringNGramsConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  sparseToDenseConfig,
  splitVConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  subConfig,
  sumConfig,
  tanhConfig,
  tileConfig,
  topKConfig,
  transformConfig,
  transposeConfig,
  unpackConfig,
  zerosLikeConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel(kernelConfig);
}
