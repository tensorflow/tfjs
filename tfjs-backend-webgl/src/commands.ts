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
import {CommandBuildOutput, DataId, DataType, env, KernelCommand, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {CPUTimerQuery, MathBackendWebGL} from './backend_webgl';
import * as gpgpu_math from './gpgpu_math';
import {GPGPUBinary, GPGPUProgram} from './gpgpu_math';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';

const WEBGL_PACKED = 'WEBGL_PACKED';
const WEBGL_UNPACKED = 'WEBGL_UNPACKED';

export class WebGLProgramCommand extends KernelCommand {
  public outDataTemplate: Partial<tex_util.TextureData> = {};
  public isOutputAlwaysEmpty: boolean = false;
  public binary?: GPGPUBinary;

  constructor(
      public backend: MathBackendWebGL, public program: GPGPUProgram,
      public customUniformValues?: number[][],
      public customTexShape?: [number, number]) {
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

  programPackTag() {
    return !!this.program.packedInputs ? WEBGL_PACKED : WEBGL_UNPACKED;
  }

  prepareOutputData() {
    // WebGLProgramCommand always has one output tensor only.
    const placeholder = this.outputs[0];
    const output = this.backend.makeTensorInfo(
        placeholder.template.shape, placeholder.template.dtype);
    const tag = !!this.program.packedOutput ? WEBGL_PACKED : WEBGL_UNPACKED;
    // Increase the ref by one since it's set with two tags.
    this.backend.incRef(output.dataId);
    placeholder.set(output, this.backend);
    placeholder.set(output, this.backend, /*tag=*/tag);

    const outData = this.backend.texData.get(placeholder.get().dataId);
    Object.assign(outData, this.outDataTemplate);
    this.backend.uploadToGPU(output.dataId);
    return {shape: output.shape, texData: outData, isUniform: false};
  }

  prepareInputsData() {
    return this.inputs.map((placeholder) => {
      let input = placeholder.getNoNotFoundError(this.programPackTag());
      let texData;
      let dataIdToDispose: DataId|null = null;

      if (input != null) {
        texData = this.backend.texData.get(input.dataId);
      } else {
        // Use the default one, upload to gpu and pack/unpack if necessary.
        input = placeholder.get();
        let texData = this.backend.texData.get(input.dataId);
        if (texData.texture == null) {
          if (!this.program.packedInputs &&
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
              uniformValues: texData.values as TypedArray
            };
          }

          // This ensures that if a packed program's inputs have not yet
          // been uploaded to the GPU, they get uploaded as packed right off
          // the bat.
          if (this.program.packedInputs) {
            texData.isPacked = true;
            texData.shape = input.shape;
          }
        }

        this.backend.uploadToGPU(input.dataId);
        if (!!texData.isPacked !== !!this.program.packedInputs) {
          const newInput = this.program.packedInputs ?
              this.backend.packTensor(input) :
              this.backend.unpackTensor(input);
          // Store the packed/unpacked tensor as additional value in
          // placeholder for reuse.
          placeholder.set(newInput, this.backend, this.programPackTag())
          texData = this.backend.texData.get(newInput.dataId);
        }
      }

      if (texData.isPacked &&
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
        const newInput = this.backend.packedReshape(input, targetShape);
        texData = this.backend.texData.get(input.dataId);
        dataIdToDispose = newInput;

        savedInput.shape = targetShape;
      }
      return {shape: input.shape, texData, isUniform: false, dataIdToDispose};
    });
  }

  runProgram(
      inputsData: ReturnType<typeof this.prepareInputsData>,
      outputData: ReturnType<typeof this.prepareOutputData>) {
    const activeTimers = this.backend.activeTimers;
    const shouldTimeProgram = activeTimers != null;
    let query: WebGLQuery|CPUTimerQuery;
    if (shouldTimeProgram) {
      query = this.backend.startTimer();
    }

    if (!env().get('ENGINE_COMPILE_ONLY')) {
      gpgpu_math.runProgram(
          this.backend.gpgpu, this.binary!, inputsData, outputData,
          this.customUniformValues);
    }

    if (shouldTimeProgram) {
      query = this.backend.endTimer(query);
      this.backend.activeTimers.push({
        name: this.program.constructor.name,
        query: this.backend.getQueryTime(query)
      });
    }
  }

  override execute() {
    const outData = this.prepareOutputData();
    if (this.isOutputAlwaysEmpty) {
      return;
    }
    const inputsData = this.prepareInputsData();
    this.runProgram(inputsData, outData);
    for (const {dataIdToDispose} of inputsData) {
      if (dataIdToDispose != null) {
        this.backend.disposeData(dataIdToDispose);
      }
    }
  }

  static override build<TensorInfo>(
      backend: MathBackendWebGL, program: GPGPUProgram, inputs: TensorInfo[],
      outputDtype: DataType, customUniformValues?: number[][],
      customTexShape?: [number, number]): CommandBuildOutput<TensorInfo> {
    const command = new WebGLProgramCommand(
        backend, program, customUniformValues, customTexShape);

    // Make output tensor
    const output = backend.makeTensorInfo(program.outputShape, outputDtype);
    command.pushOutput(output);

    // Prepare output data
    if (program.packedOutput) {
      command.outDataTemplate.isPacked = true;
    }
    if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
      const texelShape = customTexShape != null ?
          customTexShape :
          tex_util.getDenseTexShape(program.outputShape);
      // For a densely packed output, we explicitly set texShape
      // so it doesn't get assigned later according to our typical packing
      // scheme wherein a single texel can only contain values from adjacent
      // rows/cols.
      command.outDataTemplate.texShape =
          texelShape.map(d => d * 2) as [number, number];
    }
    if (program.outTexUsage != null) {
      command.outDataTemplate.usage = program.outTexUsage;
    }

    if (util.sizeFromShape(output.shape) === 0) {
      // Short-circuit the computation since the result is empty (has 0 in its
      // shape).
      command.outDataTemplate.values =
          util.getTypedArrayFromDType(output.dtype as 'float32', 0);
      command.isOutputAlwaysEmpty = true;
      const outData = backend.texData.get(output.dataId);
      Object.assign(outData, command.outDataTemplate);
      return {command, outputs: output as TensorInfo};
    }

    command.pushInput(...(inputs as any));

    // The output of record is generated by executing the command. Prepare the
    // inputs and output with tensors we have.
    for (let i = 0; i < command.inputs.length; ++i) {
      command.inputs[i].set(inputs[i] as any, backend);
    }
    command.outputs[0].set(output, backend);

    const inputsData = command.prepareInputsData();
    const outputData = command.prepareOutputData();

    const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
    const binary = backend.getAndSaveBinary(key, () => {
      return gpgpu_math.compileProgram(
          backend.gpgpu, program, inputsData, outputData);
    });
    command.binary = binary;

    command.runProgram(inputsData, outputData);

    // Cleanup input and output placeholders used to generate recording output.
    const buildOutput = command.outputs[0].release();
    for (const placeholder of command.inputs) {
      placeholder.reset();
    }

    return {command, outputs: buildOutput as any};
  }
}
