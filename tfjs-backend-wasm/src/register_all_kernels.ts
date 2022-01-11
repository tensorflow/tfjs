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

import {_fusedMatMulConfig} from './kernels/_FusedMatMul';
import {absConfig} from './kernels/Abs';
import {addConfig} from './kernels/Add';
import {addNConfig} from './kernels/AddN';
import {allConfig} from './kernels/All';
import {anyConfig} from './kernels/Any';
import {argMaxConfig} from './kernels/ArgMax';
import {avgPoolConfig} from './kernels/AvgPool';
import {batchMatMulConfig} from './kernels/BatchMatMul';
import {batchToSpaceNDConfig} from './kernels/BatchToSpaceND';
import {castConfig} from './kernels/Cast';
import {ceilConfig} from './kernels/Ceil';
import {clipByValueConfig} from './kernels/ClipByValue';
import {concatConfig} from './kernels/Concat';
import {conv2DConfig} from './kernels/Conv2D';
import {conv2DBackpropInputConfig} from './kernels/Conv2DBackpropInput';
import {cosConfig} from './kernels/Cos';
import {coshConfig} from './kernels/Cosh';
import {cropAndResizeConfig} from './kernels/CropAndResize';
import {cumsumConfig} from './kernels/Cumsum';
import {depthToSpaceConfig} from './kernels/DepthToSpace';
import {depthwiseConv2dNativeConfig} from './kernels/DepthwiseConv2dNative';
import {eluConfig} from './kernels/Elu';
import {equalConfig} from './kernels/Equal';
import {expConfig} from './kernels/Exp';
import {expandDimsConfig} from './kernels/ExpandDims';
import {fillConfig} from './kernels/Fill';
import {flipLeftRightConfig} from './kernels/FlipLeftRight';
import {floorConfig} from './kernels/Floor';
import {floorDivConfig} from './kernels/FloorDiv';
import {fusedBatchNormConfig} from './kernels/FusedBatchNorm';
import {fusedConv2DConfig} from './kernels/FusedConv2D';
import {fusedDepthwiseConv2DConfig} from './kernels/FusedDepthwiseConv2D';
import {gatherNdConfig} from './kernels/GatherNd';
import {gatherV2Config} from './kernels/GatherV2';
import {greaterConfig} from './kernels/Greater';
import {greaterEqualConfig} from './kernels/GreaterEqual';
import {identityConfig} from './kernels/Identity';
import {leakyReluConfig} from './kernels/LeakyRelu';
import {lessConfig} from './kernels/Less';
import {lessEqualConfig} from './kernels/LessEqual';
import {logConfig} from './kernels/Log';
import {logicalAndConfig} from './kernels/LogicalAnd';
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
import {rangeConfig} from './kernels/Range';
import {realDivConfig} from './kernels/RealDiv';
import {reluConfig} from './kernels/Relu';
import {relu6Config} from './kernels/Relu6';
import {reshapeConfig} from './kernels/Reshape';
import {resizeBilinearConfig} from './kernels/ResizeBilinear';
import {reverseConfig} from './kernels/Reverse';
import {rotateWithOffsetConfig} from './kernels/RotateWithOffset';
import {roundConfig} from './kernels/Round';
import {rsqrtConfig} from './kernels/Rsqrt';
import {scatterNdConfig} from './kernels/ScatterNd';
import {selectConfig} from './kernels/Select';
import {sigmoidConfig} from './kernels/Sigmoid';
import {sinConfig} from './kernels/Sin';
import {sliceConfig} from './kernels/Slice';
import {softmaxConfig} from './kernels/Softmax';
import {spaceToBatchNDConfig} from './kernels/SpaceToBatchND';
import {sparseFillEmptyRowsConfig} from './kernels/SparseFillEmptyRows';
import {sparseReshapeConfig} from './kernels/SparseReshape';
import {sparseSegmentMeanConfig} from './kernels/SparseSegmentMean';
import {sparseSegmentSumConfig} from './kernels/SparseSegmentSum';
import {splitVConfig} from './kernels/SplitV';
import {sqrtConfig} from './kernels/Sqrt';
import {squareConfig} from './kernels/Square';
import {squaredDifferenceConfig} from './kernels/SquaredDifference';
import {stepConfig} from './kernels/Step';
import {stridedSliceConfig} from './kernels/StridedSlice';
import {subConfig} from './kernels/Sub';
import {sumConfig} from './kernels/Sum';
import {tanConfig} from './kernels/Tan';
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
  allConfig,
  anyConfig,
  argMaxConfig,
  avgPoolConfig,
  batchMatMulConfig,
  batchToSpaceNDConfig,
  castConfig,
  ceilConfig,
  clipByValueConfig,
  concatConfig,
  conv2DConfig,
  conv2DBackpropInputConfig,
  cosConfig,
  coshConfig,
  cropAndResizeConfig,
  cumsumConfig,
  depthToSpaceConfig,
  depthwiseConv2dNativeConfig,
  eluConfig,
  equalConfig,
  expConfig,
  expandDimsConfig,
  fillConfig,
  flipLeftRightConfig,
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
  leakyReluConfig,
  lessConfig,
  lessEqualConfig,
  logConfig,
  logicalAndConfig,
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
  rangeConfig,
  realDivConfig,
  reluConfig,
  relu6Config,
  reshapeConfig,
  resizeBilinearConfig,
  reverseConfig,
  rotateWithOffsetConfig,
  roundConfig,
  rsqrtConfig,
  scatterNdConfig,
  selectConfig,
  sigmoidConfig,
  sinConfig,
  sliceConfig,
  softmaxConfig,
  spaceToBatchNDConfig,
  sparseFillEmptyRowsConfig,
  sparseReshapeConfig,
  sparseSegmentMeanConfig,
  sparseSegmentSumConfig,
  splitVConfig,
  sqrtConfig,
  squareConfig,
  squaredDifferenceConfig,
  stepConfig,
  stridedSliceConfig,
  subConfig,
  sumConfig,
  tanConfig,
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
