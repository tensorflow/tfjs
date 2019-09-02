/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ENV} from '../../environment';
import {DataId, registerKernel} from '../../kernel_registry';
import {kernelRegistry} from '../../kernel_registry';
import {DataType} from '../../types';
import {DataStorage, KernelBackend} from '../backend';

import {BinaryOpComplexProgram, COMPLEX_MULTIPLY} from './binaryop_complex_gpu';
import {BinaryOpProgram, MUL} from './binaryop_gpu';
import {BinaryOpPackedProgram} from './binaryop_packed_gpu';
import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';
import {TextureData} from './tex_util';

interface DataInfo {
  dtype: DataType;
  dataId: {};
  shape: number[];
}

interface WebGLStorage {
  compileAndRun(
      program: GPGPUProgram, inputs: DataInfo[], output?: DataInfo,
      customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void,
      preventEagerUnpackingOfOutput?: boolean): DataInfo;
  disposeData(dataId: DataId): void;
  makeComplexComponentTensorHandle(
      complexInfo: DataInfo, complexPart: DataInfo): DataInfo;
  texData: DataStorage<TextureData>;
  shouldExecuteOnCPU(inputs: DataInfo[], sizeThreshold?: boolean): boolean;
  cpuBackend: KernelBackend;
}

registerKernel('Mul', 'webgl', ({inputs, storage}) => {
  const {a, b} = inputs;
  const webgl = storage as WebGLStorage;
  if (a.dtype === 'complex64') {
    const aData = webgl.texData.get(a.dataId);
    const bData = webgl.texData.get(b.dataId);

    const realProgram =
        new BinaryOpComplexProgram(COMPLEX_MULTIPLY.REAL, a.shape, b.shape);
    const imagProgram =
        new BinaryOpComplexProgram(COMPLEX_MULTIPLY.IMAG, a.shape, b.shape);

    const inputs = [
      webgl.makeComplexComponentTensorHandle(a, aData.complexTensors.real),
      webgl.makeComplexComponentTensorHandle(a, aData.complexTensors.imag),
      webgl.makeComplexComponentTensorHandle(b, bData.complexTensors.real),
      webgl.makeComplexComponentTensorHandle(b, bData.complexTensors.imag)
    ];
    const real = webgl.compileAndRun(realProgram, inputs);
    const imag = webgl.compileAndRun(imagProgram, inputs);

    const complex = this.complex(real, imag);
    webgl.disposeData(real);
    webgl.disposeData(imag);
    return complex;
  }

  if (webgl.shouldExecuteOnCPU([a, b])) {
    const backend = 'cpu';
    const kernelName = 'Mul';
    const key = `${backend}_${kernelName}`;
    return kernelRegistry[key].func({inputs, storage: webgl.cpuBackend});
    // return webgl.cpuBackend.multiply(a, b);
  }
  if (ENV.getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
    return packedBinaryOp(a, b, MUL, a.dtype, webgl);
  }
  const program = new BinaryOpProgram(MUL, a.shape, b.shape);
  const output = this.makeOutputArray(program.outputShape, a.dtype);
  return webgl.compileAndRun(program, [a, b], output);
});

function packedBinaryOp(
    a: DataInfo, b: DataInfo, op: string, dtype: DataType, webgl: WebGLStorage,
    checkOutOfBounds = false) {
  const program =
      new BinaryOpPackedProgram(op, a.shape, b.shape, checkOutOfBounds);
  const output = this.makePackedTensor(program.outputShape, dtype);
  return webgl.compileAndRun(program, [a, b], output);
}
