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
import {CommandBuildOutput, DataType, env, KernelCommand, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {CPUTimerQuery, MathBackendWebGL} from './backend_webgl';
import * as gpgpu_math from './gpgpu_math';
import {GPGPUBinary, GPGPUProgram} from './gpgpu_math';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';

// const WEBGL_PACKED = 'WEBGL_PACKED';
// const WEBGL_UNPACKED = 'WEBGL_UNPACKED';

interface WebGLProgramTensorData extends gpgpu_math.TensorData {
  readonly tensorInfo: TensorInfo;
  readonly disposeTensorInfoAfterRunProgram?: boolean;
}

export class WebGLProgramCommand extends KernelCommand {
  public isOutputAlwaysEmpty = false;

  private constructor(
      public readonly backend: MathBackendWebGL,
      public readonly program: GPGPUProgram,
      public readonly outputDataTemplate: Partial<tex_util.TextureData>,
      public readonly customUniformValues?: number[][],
      public readonly customTexShape?: [number, number],
      public readonly binary?: GPGPUBinary) {
    super();
  }

  public override pushInput(...tensors: TensorInfo[]): void {
    for (const tensor of tensors) {
      if (tensor.dtype === 'complex64') {
        throw new Error(
            `GPGPUProgram does not support complex64 input. For complex64 ` +
            `dtypes, please separate the program into real and imaginary ` +
            `parts.`);
      }
    }
    super.pushInput(...tensors);
  }

  static prepareOutputData(
      backend: MathBackendWebGL, program: GPGPUProgram, outputDtype: DataType,
      outputDataTemplate: Partial<tex_util.TextureData>):
      WebGLProgramTensorData {
    const output = backend.makeTensorInfo(program.outputShape, outputDtype);
    const outData = backend.texData.get(output.dataId);

    Object.assign(outData, outputDataTemplate);
    if (outputDataTemplate.values == null) {
      backend.uploadToGPU(output.dataId);
    }

    return {
      shape: output.shape,
      texData: outData,
      isUniform: false,
      tensorInfo: output,
    };
  }

  static prepareInputData(
      backend: MathBackendWebGL, program: GPGPUProgram,
      input: TensorInfo): WebGLProgramTensorData {
    let disposeTensorInfoAfterRunProgram = false;
    let texData = backend.texData.get(input.dataId);
    if (texData.texture == null) {
      if (!program.packedInputs &&
          util.sizeFromShape(input.shape) <=
              env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
        // Upload small tensors that live on the CPU as uniforms, not as
        // textures. Do this only when the environment supports 32bit
        // floats due to problems when comparing 16bit floats with 32bit
        // floats.
        // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
        // possible for packed shaders to sample from uniforms.
        return {
          shape: input.shape,
          texData: null,
          isUniform: true,
          uniformValues: texData.values as TypedArray,
          tensorInfo: input,
        };
      }

      // This ensures that if a packed program's inputs have not yet
      // been uploaded to the GPU, they get uploaded as packed right off
      // the bat.
      if (program.packedInputs) {
        texData.isPacked = true;
        texData.shape = input.shape;
      }
    }

    backend.uploadToGPU(input.dataId);
    if (!!texData.isPacked !== !!program.packedInputs) {
      input = program.packedInputs ? backend.packTensor(input) :
                                     backend.unpackTensor(input);
      texData = backend.texData.get(input.dataId);
      disposeTensorInfoAfterRunProgram = true;
    } else if (
        texData.isPacked &&
        !webgl_util.isReshapeFree(texData.shape, input.shape)) {
      // This is a special case where a texture exists for a tensor
      // but the shapes are incompatible (due to packing constraints)
      // because the tensor did not have a chance to go through the packed
      // reshape shader. This only happens when we reshape the *same*
      // tensor to form *distinct* inputs to an op, e.g. dotting a vector
      // with itself. This case will disappear once packed uploading is
      // the default.
      const savedInput = input;
      const targetShape = input.shape;

      input.shape = texData.shape;

      input = backend.packedReshape(input, targetShape);
      texData = backend.texData.get(input.dataId);
      disposeTensorInfoAfterRunProgram = true;

      savedInput.shape = targetShape;
    }

    return {
      shape: input.shape,
      texData,
      isUniform: false,
      tensorInfo: input,
      disposeTensorInfoAfterRunProgram,
    };
  }

  static runProgram(
      backend: MathBackendWebGL, program: GPGPUProgram, binary: GPGPUBinary,
      customUniformValues: number[][]|undefined,
      inputsData: gpgpu_math.TensorData[], outputData: gpgpu_math.TensorData) {
    const activeTimers = backend.activeTimers;
    const shouldTimeProgram = activeTimers != null;
    let query: WebGLQuery|CPUTimerQuery;
    if (shouldTimeProgram) {
      query = backend.startTimer();
    }

    if (!env().get('ENGINE_COMPILE_ONLY')) {
      gpgpu_math.runProgram(
          backend.gpgpu, binary, inputsData, outputData, customUniformValues);
    }

    if (shouldTimeProgram) {
      query = backend.endTimer(query);
      backend.activeTimers.push({
        name: program.constructor.name,
        query: backend.getQueryTime(query),
      });
    }
  }

  static disposeIntermediateTensorInfos(
      backend: MathBackendWebGL, ...data: WebGLProgramTensorData[]) {
    for (const {tensorInfo, disposeTensorInfoAfterRunProgram} of data) {
      if (!!disposeTensorInfoAfterRunProgram) {
        backend.disposeData(tensorInfo.dataId);
      }
    }
  }

  override execute() {
    const cls = WebGLProgramCommand;
    const outputData = cls.prepareOutputData(
        this.backend, this.program, this.outputs[0].template.dtype,
        this.outputDataTemplate);
    this.outputs[0].set(outputData.tensorInfo);

    if (this.isOutputAlwaysEmpty) {
      return;
    }

    const inputsData = this.inputs.map((placeholder) => {
      return cls.prepareInputData(
          this.backend, this.program, placeholder.get());
    });
    cls.runProgram(
        this.backend, this.program, this.binary, this.customUniformValues,
        inputsData, outputData);
    cls.disposeIntermediateTensorInfos(this.backend, ...inputsData, outputData);
  }

  static override build<TensorInfo>(
      backend: MathBackendWebGL, program: GPGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType, customUniformValues?: number[][],
      customTexShape?: [number, number]): CommandBuildOutput<TensorInfo> {
    // Prepare output texture data template.
    const outputDataTemplate: Partial<tex_util.TextureData> = {};
    if (program.packedOutput) {
      outputDataTemplate.isPacked = true;
    }
    if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
      const texelShape = customTexShape != null ?
          customTexShape :
          tex_util.getDenseTexShape(program.outputShape);
      // For a densely packed output, we explicitly set texShape
      // so it doesn't get assigned later according to our typical packing
      // scheme wherein a single texel can only contain values from adjacent
      // rows/cols.
      outputDataTemplate.texShape =
          texelShape.map(d => d * 2) as [number, number];
    }
    if (program.outTexUsage != null) {
      outputDataTemplate.usage = program.outTexUsage;
    }

    let isOutputAlwaysEmpty = false;
    if (util.sizeFromShape(program.outputShape) === 0) {
      outputDataTemplate.values =
          util.getTypedArrayFromDType(outputDtype as 'float32', 0);
      isOutputAlwaysEmpty = true;
    }

    const outputData = this.prepareOutputData(
        backend, program, outputDtype, outputDataTemplate);
    if (isOutputAlwaysEmpty) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      let command = undefined;
      if (!this.noCommandRecording) {
        command = new WebGLProgramCommand(
            backend, program, outputDataTemplate, customUniformValues,
            customTexShape);
        command.isOutputAlwaysEmpty = true;
      }
      return {command, outputs: outputData.tensorInfo as TensorInfo};
    }

    const inputsData = inputs.map((input: TensorInfo) => {
      return this.prepareInputData(backend, program, input as any);
    });

    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = backend.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          backend.gpgpu, program, inputsData, outputData);
    });

    this.runProgram(
        backend, program, binary, customUniformValues, inputsData, outputData);
    this.disposeIntermediateTensorInfos(backend, ...inputsData, outputData);

    let command: WebGLProgramCommand|undefined = undefined;
    if (!this.noCommandRecording()) {
      // Build the Command and TensorPLaceholder objects only when needed.
      command = new WebGLProgramCommand(
          backend, program, outputDataTemplate, customUniformValues,
          customTexShape, binary);
      command.pushInput(...(inputs as any));
      command.pushOutput(outputData.tensorInfo);
    }

    return {command, outputs: outputData.tensorInfo as any};
  }
}
