import {conv2d as conv2d_fused} from '@tensorflow/tfjs-core/dist/ops/fused/conv2d';
export const fused = {
	conv2d: conv2d_fused,
};
export {depthwiseConv2d} from '@tensorflow/tfjs-core/dist/ops/depthwise_conv2d';
export {add} from '@tensorflow/tfjs-core/dist/ops/add';
export {relu} from '@tensorflow/tfjs-core/dist/ops/relu';
export {pad} from '@tensorflow/tfjs-core/dist/ops/pad';
export {maxPool} from '@tensorflow/tfjs-core/dist/ops/max_pool';
export {tensor1d} from '@tensorflow/tfjs-core/dist/ops/tensor1d';
export {stridedSlice} from '@tensorflow/tfjs-core/dist/ops/strided_slice';
export {squeeze} from '@tensorflow/tfjs-core/dist/ops/squeeze';
export {reshape} from '@tensorflow/tfjs-core/dist/ops/reshape';
export {stack} from '@tensorflow/tfjs-core/dist/ops/stack';
export {concat} from '@tensorflow/tfjs-core/dist/ops/concat';