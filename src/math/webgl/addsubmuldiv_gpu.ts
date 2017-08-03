/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import {MatrixOrientation} from '../math';

import * as binaryop_gpu from './binaryop_gpu';
import {GPGPUContext} from './gpgpu_context';

export type Operation = '+' | '-' | '*' | '/';

export enum OperandType {
  MATRIX,
  SCALAR
}

export function getFragmentShaderSource(
    aType: OperandType, aOrientation: MatrixOrientation, op: Operation,
    bType: OperandType, bOrientation: MatrixOrientation): string {
  const aUV = operandToShaderSnippet(aType, aOrientation);
  const bUV = operandToShaderSnippet(bType, bOrientation);
  const resultOp = `gl_FragColor = vec4(a ${op} b, 0, 0, 0);`;
  return binaryop_gpu.getFragmentShaderSource(aUV, bUV, resultOp);
}

function operandToShaderSnippet(
    operand: OperandType, orientation: MatrixOrientation): string {
  switch (operand) {
    case OperandType.MATRIX:
      return 'resultUV' +
          (orientation === MatrixOrientation.REGULAR ? '.st' : '.ts');
    case OperandType.SCALAR:
      return 'vec2(0.5, 0.5)';
    default:
      throw new Error('Unknown operand type');
  }
}

export function addSubMulDiv(
    gpgpu: GPGPUContext, program: WebGLProgram, a: WebGLTexture,
    aShapeRowCol: [number, number], b: WebGLTexture,
    bShapeRowCol: [number, number], result: WebGLTexture,
    resultShapeRowCol: [number, number]) {
  return binaryop_gpu.binaryOp(
      gpgpu, program, a, aShapeRowCol, b, bShapeRowCol, result,
      resultShapeRowCol);
}

export function uploadScalarPlusMatrixDownload(
    a: number, b: Float32Array, bShape: [number, number],
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.SCALAR, MatrixOrientation.REGULAR, '+', OperandType.MATRIX,
      bOrientation);
  return binaryop_gpu.uploadBinaryOpDownload(
      new Float32Array([a]), [1, 1], b, bShape, src);
}

export function uploadMatrixMinusScalarDownload(
    a: Float32Array, aShape: [number, number], b: number,
    aOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.MATRIX, aOrientation, '-', OperandType.SCALAR,
      MatrixOrientation.REGULAR);
  return binaryop_gpu.uploadBinaryOpDownload(
      a, aShape, new Float32Array([b]), [1, 1], src);
}

export function uploadScalarMinusMatrixDownload(
    a: number, b: Float32Array, bShape: [number, number],
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.SCALAR, MatrixOrientation.REGULAR, '-', OperandType.MATRIX,
      bOrientation);
  return binaryop_gpu.uploadBinaryOpDownload(
      new Float32Array([a]), [1, 1], b, bShape, src);
}

export function uploadScalarTimesMatrixDownload(
    a: number, b: Float32Array, bShape: [number, number],
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.SCALAR, MatrixOrientation.REGULAR, '*', OperandType.MATRIX,
      bOrientation);
  return binaryop_gpu.uploadBinaryOpDownload(
      new Float32Array([a]), [1, 1], b, bShape, src);
}

export function uploadMatrixTimesMatrixDownload(
    a: Float32Array, b: Float32Array, shape: [number, number],
    aOrientation = MatrixOrientation.REGULAR,
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.MATRIX, aOrientation, '*', OperandType.MATRIX, bOrientation);
  return binaryop_gpu.uploadBinaryOpDownload(a, shape, b, shape, src);
}

export function uploadMatrixPlusMatrixDownload(
    a: Float32Array, b: Float32Array, shape: [number, number],
    aOrientation = MatrixOrientation.REGULAR,
    bOrientation = MatrixOrientation.REGULAR): Float32Array {
  const src = getFragmentShaderSource(
      OperandType.MATRIX, aOrientation, '+', OperandType.MATRIX, bOrientation);
  return binaryop_gpu.uploadBinaryOpDownload(a, shape, b, shape, src);
}
