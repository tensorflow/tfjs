// conv3dDerFilter(x: Tensor5D, dy: Tensor5D, convInfo:
// backend_util.Conv3DInfo):
//       Tensor5D {
//     const program = new Conv3DDerFilterProgram(convInfo);
//     return this.compileAndRun(program, [x, dy]);
//   }
