// depthwiseConv2D(
//   x: Tensor4D, filter: Tensor4D,
//   convInfo: backend_util.Conv2DInfo): Tensor4D {
// let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
// if (env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
//     convInfo.strideWidth <= 2 &&
//     convInfo.outChannels / convInfo.inChannels === 1) {
//   program = new DepthwiseConvPacked2DProgram(convInfo);
//   return this.compileAndRun(program, [x, filter]);
// }

// program = new DepthwiseConv2DProgram(convInfo);
// return this.compileAndRun(program, [x, filter]);
// }
