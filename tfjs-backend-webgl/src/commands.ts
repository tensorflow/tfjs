/**
 * @license
 * Copyright 2023 Google LLC.
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
import {ClosureCommand, CommandBuildOutput, DataType, KernelCommand, TensorInfo, util} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from './backend_webgl';
import {getUniformLocations, GPGPUBinary, GPGPUProgram, TensorData} from './gpgpu_math';
import * as tex_util from './tex_util';

export class RunWebGLProgramCommand extends KernelCommand {
  static override build(
      backend: MathBackendWebGL, program: GPGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType, customUniformValues?: number[][],
      customTexShape?: [number, number]):
      CommandBuildOutput<RunWebGLProgramCommand, TensorInfo> {
    const output = backend.makeTensorInfo(program.outputShape, outputDtype);
    const outData = backend.texData.get(output.dataId);
    if (program.packedOutput) {
      outData.isPacked = true;
    }
    if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
      const texelShape = customTexShape != null ?
          customTexShape :
          tex_util.getDenseTexShape(program.outputShape);
      // For a densely packed output, we explicitly set texShape
      // so it doesn't get assigned later according to our typical packing
      // scheme wherein a single texel can only contain values from adjacent
      // rows/cols.
      outData.texShape = texelShape.map(d => d * 2) as [number, number];
    }
    if (program.outTexUsage != null) {
      outData.usage = program.outTexUsage;
    }

    if (util.sizeFromShape(output.shape) === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      outData.values =
          util.getTypedArrayFromDType(output.dtype as 'float32', 0);
      return output;
    }

    const dataToDispose: TensorInfo[] = [];
    const inputsData: TensorData[] = inputs.map(input => {
      if (input.dtype === 'complex64') {
        throw new Error(
            `GPGPUProgram does not support complex64 input. For complex64 ` +
            `dtypes, please separate the program into real and imaginary ` +
            `parts.`);
      }

      let texData = this.texData.get(input.dataId);

      if (texData.texture == null) {
        if (!program.packedInputs &&
            util.sizeFromShape(input.shape) <=
                env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
          // Upload small tensors that live on the CPU as uniforms, not as
          // textures. Do this only when the environment supports 32bit floats
          // due to problems when comparing 16bit floats with 32bit floats.
          // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
          // possible for packed shaders to sample from uniforms.
          return {
            shape: input.shape,
            texData: null,
            isUniform: true,
            uniformValues: texData.values as TypedArray
          };
        }

        // This ensures that if a packed program's inputs have not yet been
        // uploaded to the GPU, they get uploaded as packed right off the bat.
        if (program.packedInputs) {
          texData.isPacked = true;
          texData.shape = input.shape;
        }
      }

      this.uploadToGPU(input.dataId);
      if (!!texData.isPacked !== !!program.packedInputs) {
        input = texData.isPacked ? this.unpackTensor(input) :
                                   this.packTensor(input);
        dataToDispose.push(input);
        texData = this.texData.get(input.dataId);
      } else if (
          texData.isPacked &&
          !webgl_util.isReshapeFree(texData.shape, input.shape)) {
        // This is a special case where a texture exists for a tensor
        // but the shapes are incompatible (due to packing constraints) because
        // the tensor did not have a chance to go through the packed reshape
        // shader. This only happens when we reshape the *same* tensor to form
        // *distinct* inputs to an op, e.g. dotting a vector with itself. This
        // case will disappear once packed uploading is the default.

        const savedInput = input;
        const targetShape = input.shape;

        input.shape = texData.shape;
        input = this.packedReshape(input as Tensor, targetShape);
        dataToDispose.push(input);
        texData = this.texData.get(input.dataId);

        savedInput.shape = targetShape;
      }

      return {shape: input.shape, texData, isUniform: false};
    });

    backend.uploadToGPU(output.dataId);
    const outputData:
        TensorData = {shape: output.shape, texData: outData, isUniform: false};
    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = override.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          this.gpgpu, program, inputsData, outputData);
    });
    const shouldTimeProgram = this.activeTimers != null;
    let query: WebGLQuery|CPUTimerQuery;
    if (shouldTimeProgram) {
      query = this.startTimer();
    }

    if (!env().get('ENGINE_COMPILE_ONLY')) {
      gpgpu_math.runProgram(
          this.gpgpu, binary, inputsData, outputData, customUniformValues);
    }
  }

  execute() {}
}
