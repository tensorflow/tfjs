// conv3d(x: Tensor5D, filter: Tensor5D, convInfo: backend_util.Conv3DInfo):
//       Tensor5D {
//     const program = new Conv3DProgram(convInfo);
//     return this.compileAndRun(program, [x, filter]);
//   }
