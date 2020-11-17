// fusedDepthwiseConv2D(
//   {input, filter, convInfo, bias, activation, preluActivationWeights}:
//       backend_util.FusedConv2DConfig): Tensor4D {
// const shouldPackDepthwiseConv = env().getBool('WEBGL_PACK_DEPTHWISECONV') &&
//     convInfo.strideWidth <= 2 &&
//     convInfo.outChannels / convInfo.inChannels === 1;
// const fusedActivation = activation ?
//     mapActivationToShaderProgram(activation, shouldPackDepthwiseConv) :
//     null;
// const inputs: Tensor[] = [input, filter];

// const hasBias = bias != null;
// const hasPreluActivationWeights = preluActivationWeights != null;
// if (hasBias) {
//   inputs.push(bias);
// }
// if (hasPreluActivationWeights) {
//   inputs.push(preluActivationWeights);
// }

// let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
// if (shouldPackDepthwiseConv) {
//   program = new DepthwiseConvPacked2DProgram(
//       convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
//   return this.compileAndRun(program, inputs);
// }

// program = new DepthwiseConv2DProgram(
//     convInfo, hasBias, fusedActivation, hasPreluActivationWeights);
// return this.compileAndRun(program, inputs);
// }
