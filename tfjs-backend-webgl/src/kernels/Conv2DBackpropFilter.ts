// import {Conv2DDerFilterProgram, Conv2DDerInputProgram,
// Conv3DDerFilterProgram, Conv3DDerInputProgram} from './conv_backprop_gpu';
// import {DepthwiseConv2DDerFilterProgram, DepthwiseConv2DDerInputProgram} from
// './conv_backprop_gpu_depthwise'; import {Conv3DProgram} from './conv_gpu';
// import {DepthwiseConv2DProgram} from './conv_gpu_depthwise';
// import {DepthwiseConvPacked2DProgram} from './conv_packed_gpu_depthwise';

// conv2dDerFilter(x: Tensor4D, dy: Tensor4D, convInfo:
// backend_util.Conv2DInfo):
//       Tensor4D {
//     const program = new Conv2DDerFilterProgram(convInfo);
//     return this.compileAndRun(program, [x, dy]);
//   }
