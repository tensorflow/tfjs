/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import {backend_util, env, Tensor, TypedArray, util} from '@tensorflow/tfjs-core';

import {GPGPUContext} from './gpgpu_context';
import * as shader_compiler from './shader_compiler';
import {InputInfo, ShapeInfo, UniformType} from './shader_compiler';
import {PackingScheme, TextureData, TextureUsage} from './tex_util';

export interface GPGPUProgram {
  variableNames: string[];
  outputShape: number[];
  userCode: string;
  enableShapeUniforms?: boolean;
  /** If true, this program expects packed input textures. Defaults to false. */
  packedInputs?: boolean;
  /** If true, this program produces a packed texture. Defaults to false. */
  packedOutput?: boolean;
  /**
   * Affects what type of texture we allocate for the output. Defaults to
   * `TextureUsage.RENDER`.
   */
  outTexUsage?: TextureUsage;
  /**
   * The type of scheme to use when packing texels for the output values.
   * See `PackingScheme` for details. Defaults to `PackingScheme.SHARED_BATCH`.
   */
  outPackingScheme?: PackingScheme;
  customUniforms?:
      Array<{name: string; arrayIndex?: number; type: UniformType;}>;
}

export interface GPGPUBinary {
  webGLProgram: WebGLProgram;
  program: GPGPUProgram;
  uniformLocations: {[name: string]: WebGLUniformLocation};
  customUniformLocations?: WebGLUniformLocation[];
  source: string;
  inShapeInfos: ShapeInfo[];
  outShapeInfo: ShapeInfo;
  infLoc: WebGLUniformLocation;
  nanLoc: WebGLUniformLocation;
  inShapesLocations?: {[name: string]: WebGLUniformLocation};
  inTexShapesLocations?: {[name: string]: WebGLUniformLocation};
  outShapeLocation?: WebGLUniformLocation;
  outShapeStridesLocation?: WebGLUniformLocation;
  outTexShapeLocation?: WebGLUniformLocation;
}

export interface TensorData {
  shape: number[];
  texData: TextureData;
  isUniform: boolean;
  // Available when we decide to upload as uniform instead of texture.
  uniformValues?: TypedArray;
}

export function compileProgram<T extends Tensor, K extends Tensor>(
    gpgpu: GPGPUContext, program: GPGPUProgram, inputs: TensorData[],
    output: TensorData): GPGPUBinary {
  const inputInfos: InputInfo[] = inputs.map((input, i) => {
    const shapeInfo: ShapeInfo = {
      logicalShape: input.shape,
      texShape: input.isUniform ? null : input.texData.texShape,
      isUniform: input.isUniform,
      isPacked: input.isUniform ? false : input.texData.isPacked,
      flatOffset: null
    };
    if (input.texData != null && input.texData.slice != null &&
        input.texData.slice.flatOffset > 0) {
      shapeInfo.flatOffset = input.texData.slice.flatOffset;
    }
    return {name: program.variableNames[i], shapeInfo};
  });
  const inShapeInfos = inputInfos.map(x => x.shapeInfo);
  const outShapeInfo: ShapeInfo = {
    logicalShape: output.shape,
    texShape: output.texData.texShape,
    isUniform: false,
    isPacked: output.texData.isPacked,
    flatOffset: null
  };
  const source = shader_compiler.makeShader(inputInfos, outShapeInfo, program);

  const webGLProgram = gpgpu.createProgram(source);

  // Add special uniforms (NAN, INFINITY)
  let infLoc: WebGLUniformLocation = null;
  const nanLoc = gpgpu.getUniformLocation(webGLProgram, 'NAN', false);
  if (env().getNumber('WEBGL_VERSION') === 1) {
    infLoc = gpgpu.getUniformLocation(webGLProgram, 'INFINITY', false);
  }

  // Add user-defined uniforms
  const shouldThrow = false;
  const uniformLocations: {[name: string]: WebGLUniformLocation} = {};
  const inShapesLocations: {[name: string]: WebGLUniformLocation} = {};
  const inTexShapesLocations: {[name: string]: WebGLUniformLocation} = {};
  for (let i = 0; i < program.variableNames.length; i++) {
    const varName = program.variableNames[i];
    uniformLocations[varName] =
        gpgpu.getUniformLocation(webGLProgram, varName, shouldThrow);
    uniformLocations[`offset${varName}`] =
        gpgpu.getUniformLocation(webGLProgram, `offset${varName}`, shouldThrow);
    if (program.enableShapeUniforms) {
      inShapesLocations[`${varName}Shape`] = gpgpu.getUniformLocation(
          webGLProgram, `${varName}Shape`, shouldThrow);
      inTexShapesLocations[`${varName}TexShape`] = gpgpu.getUniformLocation(
          webGLProgram, `${varName}TexShape`, shouldThrow);
    }
  }

  let outShapeLocation: WebGLUniformLocation;
  let outTexShapeLocation: WebGLUniformLocation;
  let outShapeStridesLocation: WebGLUniformLocation;
  if (program.enableShapeUniforms) {
    outShapeLocation =
        gpgpu.getUniformLocation(webGLProgram, 'outShape', shouldThrow);
    outShapeStridesLocation =
        gpgpu.getUniformLocation(webGLProgram, 'outShapeStrides', shouldThrow);
    outTexShapeLocation =
        gpgpu.getUniformLocation(webGLProgram, 'outTexShape', shouldThrow);
  }

  const customUniformLocations: WebGLUniformLocation[] = [];
  if (program.customUniforms) {
    program.customUniforms.forEach((d, i) => {
      customUniformLocations[i] =
          gpgpu.getUniformLocation(webGLProgram, d.name, shouldThrow);
    });
  }

  return {
    program,
    source,
    webGLProgram,
    uniformLocations,
    customUniformLocations,
    inShapeInfos,
    outShapeInfo,
    infLoc,
    nanLoc,
    inShapesLocations,
    inTexShapesLocations,
    outShapeLocation,
    outShapeStridesLocation,
    outTexShapeLocation
  };
}

function validateBinaryAndProgram(
    shapeInfos: ShapeInfo[], inputs: TensorData[]) {
  if (shapeInfos.length !== inputs.length) {
    throw Error(
        `Binary was compiled with ${shapeInfos.length} inputs, but ` +
        `was executed with ${inputs.length} inputs`);
  }

  shapeInfos.forEach((s, i) => {
    const shapeA = s.logicalShape;
    const input = inputs[i];
    const shapeB = input.shape;

    if (!util.arraysEqual(shapeA, shapeB)) {
      throw Error(
          `Binary was compiled with different shapes than ` +
          `the current args. Shapes ${shapeA} and ${shapeB} must match`);
    }
    // The input is uploaded as uniform.
    if (s.isUniform && input.isUniform) {
      return;
    }

    const texShapeA = s.texShape;
    const texShapeB = input.isUniform ? null : input.texData.texShape;
    if (!util.arraysEqual(texShapeA, texShapeB)) {
      throw Error(
          `Binary was compiled with different texture shapes than the` +
          ` current args. Shape ${texShapeA} and ${texShapeB} must match`);
    }
  });
}

export function runProgram<T extends Tensor, K extends Tensor>(
    gpgpu: GPGPUContext, binary: GPGPUBinary, inputs: TensorData[],
    output: TensorData, customUniformValues?: number[][]): void {
  if (!binary.program.enableShapeUniforms) {
    validateBinaryAndProgram(binary.inShapeInfos, inputs);
    validateBinaryAndProgram([binary.outShapeInfo], [output]);
  }

  const outTex = output.texData.texture;
  const outTexShape = output.texData.texShape;
  if (output.texData.isPacked) {
    gpgpu.setOutputPackedMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  } else {
    gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
  }
  gpgpu.setProgram(binary.webGLProgram);

  // Set special uniforms (NAN, INFINITY)
  if (env().getNumber('WEBGL_VERSION') === 1) {
    if (binary.infLoc !== null) {
      gpgpu.gl.uniform1f(binary.infLoc, Infinity);
    }
  }
  if (binary.nanLoc !== null) {
    gpgpu.gl.uniform1f(binary.nanLoc, NaN);
  }

  // Set user-defined inputs
  inputs.forEach((input, i) => {
    const varName = binary.program.variableNames[i];
    const varLoc = binary.uniformLocations[varName];
    const varOffsetLoc = binary.uniformLocations[`offset${varName}`];
    const varShapeLoc = binary.inShapesLocations[`${varName}Shape`];
    const varTexShapeLoc = binary.inTexShapesLocations[`${varName}TexShape`];

    if (varShapeLoc) {
      const {uniformShape} = shader_compiler.getUniformInfoFromShape(
          binary.program.packedInputs, input.shape, input.texData.texShape);
      switch (uniformShape.length) {
        case 1:
          gpgpu.gl.uniform1iv(varShapeLoc, new Int32Array(uniformShape));
          break;
        case 2:
          gpgpu.gl.uniform2iv(varShapeLoc, new Int32Array(uniformShape));
          break;
        case 3:
          gpgpu.gl.uniform3iv(varShapeLoc, new Int32Array(uniformShape));
          break;
        case 4:
          gpgpu.gl.uniform4iv(varShapeLoc, new Int32Array(uniformShape));
          break;
        default:
          break;
      }
    }
    if (varTexShapeLoc) {
      gpgpu.gl.uniform2i(
          varTexShapeLoc, input.texData.texShape[0], input.texData.texShape[1]);
    }

    if (varLoc == null) {
      // The compiler inferred that this variable is not used in this shader.
      return;
    }

    if (input.isUniform) {
      // Upload the values of the tensor as uniform.
      if (util.sizeFromShape(input.shape) < 2) {
        gpgpu.gl.uniform1f(varLoc, input.uniformValues[0]);
      } else {
        let vals = input.uniformValues;
        if (!(vals instanceof Float32Array)) {
          vals = new Float32Array(vals);
        }
        gpgpu.gl.uniform1fv(varLoc, vals);
      }
      return;
    }

    // If the input was sliced, upload the flat offset index.
    if (input.texData.slice != null && varOffsetLoc != null) {
      gpgpu.gl.uniform1i(varOffsetLoc, input.texData.slice.flatOffset);
    }

    gpgpu.setInputMatrixTexture(input.texData.texture, varLoc, i);
  });

  const outShapeLoc = binary.outShapeLocation;
  if (outShapeLoc) {
    switch (output.shape.length) {
      case 1:
        gpgpu.gl.uniform1iv(outShapeLoc, new Int32Array(output.shape));
        break;
      case 2:
        gpgpu.gl.uniform2iv(outShapeLoc, new Int32Array(output.shape));
        break;
      case 3:
        gpgpu.gl.uniform3iv(outShapeLoc, new Int32Array(output.shape));
        break;
      case 4:
        gpgpu.gl.uniform4iv(outShapeLoc, new Int32Array(output.shape));
        break;
      default:
        break;
    }
  }
  if (binary.outShapeStridesLocation) {
    const strides = util.computeStrides(output.shape);
    switch (output.shape.length) {
      case 2:
        gpgpu.gl.uniform1iv(
            binary.outShapeStridesLocation, new Int32Array(strides));
        break;
      case 3:
        gpgpu.gl.uniform2iv(
            binary.outShapeStridesLocation, new Int32Array(strides));
        break;
      case 4:
        gpgpu.gl.uniform3iv(
            binary.outShapeStridesLocation, new Int32Array(strides));
        break;
      default:
        break;
    }
  }
  if (binary.outTexShapeLocation) {
    gpgpu.gl.uniform2i(
        binary.outTexShapeLocation, output.texData.texShape[0],
        output.texData.texShape[1]);
  }

  if (binary.program.customUniforms && customUniformValues) {
    binary.program.customUniforms.forEach((d, i) => {
      const customLoc = binary.customUniformLocations[i];
      const customValue = customUniformValues[i];
      if (d.type === 'float') {
        gpgpu.gl.uniform1fv(customLoc, customValue);
      } else if (d.type === 'vec2') {
        gpgpu.gl.uniform2fv(customLoc, customValue);
      } else if (d.type === 'vec3') {
        gpgpu.gl.uniform3fv(customLoc, customValue);
      } else if (d.type === 'vec4') {
        gpgpu.gl.uniform4fv(customLoc, customValue);
      } else if (d.type === 'int') {
        gpgpu.gl.uniform1iv(customLoc, customValue);
      } else if (d.type === 'ivec2') {
        gpgpu.gl.uniform2iv(customLoc, customValue);
      } else if (d.type === 'ivec3') {
        gpgpu.gl.uniform3iv(customLoc, customValue);
      } else if (d.type === 'ivec4') {
        gpgpu.gl.uniform4iv(customLoc, customValue);
      } else {
        throw Error(`uniform type ${d.type} is not supported yet.`);
      }
    });
  }
  gpgpu.executeProgram();
}

export function makeShaderKey(
    program: GPGPUProgram, inputs: TensorData[], output: TensorData): string {
  let keyInputs = '';
  inputs.concat(output).forEach(x => {
    const hasOffset = x.texData != null && x.texData.slice != null &&
        x.texData.slice.flatOffset > 0;
    // TODO: Remove the condition of !x.isUniform.
    if (program.enableShapeUniforms && !x.isUniform) {
      const xTexShape = x.texData.texShape;
      const {useSqueezeShape, uniformShape, keptDims} =
          shader_compiler.getUniformInfoFromShape(
              program.packedInputs, x.shape, xTexShape);
      let rank1 = '', rank2 = '', rank34 = '';
      if (uniformShape.length === 1 && program.packedInputs) {
        const packedTexShape =
            [Math.ceil(xTexShape[0] / 2), Math.ceil(xTexShape[1] / 2)];
        rank1 = `${packedTexShape[0] > 1}_${packedTexShape[1] > 1}`;
      } else if (uniformShape.length === 2 && !program.packedInputs) {
        rank2 = `${uniformShape[0] > 1}_${uniformShape[1] > 1}`;
      } else if (uniformShape.length > 2 && !program.packedInputs) {
        const strides = util.computeStrides(uniformShape);
        rank34 = `${strides[0] === xTexShape[1]}_${
            strides[strides.length - 1] === xTexShape[1]}`;
      }
      const xRank = x.shape.length;
      const isLogicalShapTexShapeEqual =
          uniformShape.length === 2 && util.arraysEqual(x.shape, xTexShape);
      const isScalar = util.sizeFromShape(x.shape) === 1;
      const broadcastDims =
          backend_util.getBroadcastDims(x.shape, output.shape);
      const isInOutTexShapeEqual = !program.packedInputs &&
          xRank === output.shape.length &&
          util.arraysEqual(xTexShape, output.texData.texShape);
      const isTexShapeGreaterThanOne =
          program.packedInputs || uniformShape.length > 2 ?
          '' :
          `${xTexShape[0] > 1}_${xTexShape[1] > 1}`;
      // These key components are needed due to shader_compiler is embedding
      // them in the shader.
      // |xRank| is used to determine the coords length. See
      // get[Packed]SamplerAtOutputCoords.
      // |isInOutTexShapeEqual| is used to determine whether going to an
      // optimization path in getSamplerAtOutputCoords.
      // |useSqueezeShape| is extracted from squeezeInputInfo of
      // getSampler[2|3|4]D/getPackedSampler3D.
      // |isScalar| is extracted from isInputScalar/isOutputScalar in
      // getPackedSamplerAtOutputCoords.
      // |broadcastDims| is extracted from get[Packed]SamplerAtOutputCoords.
      // |isLogicalShapTexShapeEqual| is used in
      // getOutput[Packed]2DCoords/get[Packed]Sampler2D.
      // |rank1| is used in getOutputPacked1DCoords.
      // |rank2| is used in getOutput2DCoords.
      // |rank34| is used in getSampler3D/getSampler4D.
      // |isTexShapeGreaterThanOne| are used in
      // getSampler[Scalar|1D|2D]/getOutput1DCoords.
      keyInputs += `${xRank}_${isInOutTexShapeEqual}_${
          useSqueezeShape ? keptDims : ''}_${uniformShape.length}_${isScalar}_${
          broadcastDims}_${isLogicalShapTexShapeEqual}_${rank1}_${rank2}_${
          rank34}_${isTexShapeGreaterThanOne}_${hasOffset}`;
    } else {
      const texShape = x.isUniform ? 'uniform' : x.texData.texShape;
      keyInputs += `${x.shape}_${texShape}_${hasOffset}`;
    }
  });
  const keyUserCode = program.userCode;
  let key = program.constructor.name;
  // Fast string concat. See https://jsperf.com/string-concatenation/14.
  key += '_' + keyInputs + '_' + keyUserCode +
      `${env().getNumber('WEBGL_VERSION')}`;
  return key;
}

export function useShapeUniforms(rank: number) {
  // TODO: Remove the limitaion of rank <= 4.
  return env().getBool('WEBGL_USE_SHAPES_UNIFORMS') && rank <= 4;
}
