/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {MatrixOrientation} from '../types/matmul';

import {GPGPUContext} from './gpgpu_context';
import * as webgl_util from './webgl_util';

export function getFragmentShaderSource(
    sharedDimension: number, aOrientation: MatrixOrientation,
    bOrientation: MatrixOrientation): string {
  /*
      A = [0 1   B = [0 1  out = [A0*B0+A1*B2 A0*B1+A1*B3
           2 3]       2 3]        A2*B0+A1*B2 A2*B1+Aw*B3]
      out.0 = A0 * B0 + A1 * B2
      out.1 = A0 * B1 + A1 * B3
      out.2 = A2 * B0 + A3 * B2
      out.3 = A2 * B1 + A3 * B3

      A*B     = A.xxzz * B.xyxy + A.yyww * B.zwzw
      A^t*B   = A.xxyy * B.xyxy + A.zzww * B.zwzw
      A*B^t   = A.xxzz * B.xzxz + A.yyww * B.ywyw
      A^t*B^t = A.xxyy * B.xzxz + A.zzww * B.ywyw
   */
  const sharedDimensionPacked = Math.ceil(sharedDimension / 2);
  const aSample = (aOrientation === MatrixOrientation.REGULAR) ?
      'center, resultUV.t' :
      'resultUV.t, center';
  const bSample = (bOrientation === MatrixOrientation.REGULAR) ?
      'resultUV.s, center' :
      'center, resultUV.s';
  const aSwizzle: [string, string] =
      (aOrientation === MatrixOrientation.REGULAR) ? ['a.xxzz', 'a.yyww'] :
                                                     ['a.xxyy', 'a.zzww'];
  const bSwizzle: [string, string] =
      (bOrientation === MatrixOrientation.REGULAR) ? ['b.xyxy', 'b.zwzw'] :
                                                     ['b.xzxz', 'b.ywyw'];
  return `
    precision highp float;
    uniform sampler2D matrixA;
    uniform sampler2D matrixB;
    varying vec2 resultUV;

    const float sharedDimension = ${sharedDimensionPacked}.0;

    vec4 dot2x2ARowBCol() {
      vec4 result = vec4(0, 0, 0, 0);
      for (int ii = 0; ii < ${sharedDimensionPacked}; ii++) {
        float i = float(ii);
        float center = (i + 0.5) / sharedDimension;
        vec4 a = texture2D(matrixA, vec2(${aSample}));
        vec4 b = texture2D(matrixB, vec2(${bSample}));
        result +=
          (${aSwizzle[0]} * ${bSwizzle[0]}) + (${aSwizzle[1]} * ${bSwizzle[1]});
      }
      return result;
    }

    void main() {
      gl_FragColor = dot2x2ARowBCol();
    }`;
}

export function multiplyMatrixPacked(
    gpgpu: GPGPUContext, multiplyProgram: WebGLProgram, a: WebGLTexture,
    b: WebGLTexture, result: WebGLTexture,
    resultShapeRowCol: [number, number]) {
  gpgpu.setOutputPackedMatrixTexture(
      result, resultShapeRowCol[0], resultShapeRowCol[1]);
  gpgpu.setProgram(multiplyProgram);
  const matrixASamplerLocation = webgl_util.getProgramUniformLocationOrThrow(
      gpgpu.gl, multiplyProgram, 'matrixA');
  const matrixBSamplerLocation = webgl_util.getProgramUniformLocationOrThrow(
      gpgpu.gl, multiplyProgram, 'matrixB');
  gpgpu.setInputMatrixTexture(a, matrixASamplerLocation, 0);
  gpgpu.setInputMatrixTexture(b, matrixBSamplerLocation, 1);
  gpgpu.executeProgram();
}

export function uploadMultiplyMatrixPackedDownload(
    a: Float32Array, aShapeRowCol: [number, number], b: Float32Array,
    bShapeRowCol: [number, number], aOrientation = MatrixOrientation.REGULAR,
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const resultNumRows = (aOrientation === MatrixOrientation.REGULAR) ?
      aShapeRowCol[0] :
      aShapeRowCol[1];
  const resultNumCols = (bOrientation === MatrixOrientation.REGULAR) ?
      bShapeRowCol[1] :
      bShapeRowCol[0];
  const sharedDimension = (aOrientation === MatrixOrientation.REGULAR) ?
      aShapeRowCol[1] :
      aShapeRowCol[0];

  const gpgpu = new GPGPUContext();
  const program: WebGLProgram = gpgpu.createProgram(
      getFragmentShaderSource(sharedDimension, aOrientation, bOrientation));

  const aTexture: WebGLTexture =
      gpgpu.createPackedMatrixTexture(aShapeRowCol[0], aShapeRowCol[1]);
  const bTexture: WebGLTexture =
      gpgpu.createPackedMatrixTexture(bShapeRowCol[0], bShapeRowCol[1]);
  const resultTexture: WebGLTexture =
      gpgpu.createPackedMatrixTexture(resultNumRows, resultNumCols);

  gpgpu.uploadMatrixToPackedTexture(
      aTexture, aShapeRowCol[0], aShapeRowCol[1], a);
  gpgpu.uploadMatrixToPackedTexture(
      bTexture, bShapeRowCol[0], bShapeRowCol[1], b);

  multiplyMatrixPacked(
      gpgpu, program, aTexture, bTexture, resultTexture,
      [resultNumRows, resultNumCols]);

  const result = gpgpu.downloadMatrixFromPackedTexture(
      resultTexture, resultNumRows, resultNumCols);

  gpgpu.deleteMatrixTexture(aTexture);
  gpgpu.deleteMatrixTexture(bTexture);
  gpgpu.deleteMatrixTexture(resultTexture);
  gpgpu.deleteProgram(program);
  gpgpu.dispose();

  return result;
}
