/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {DataType, TypedArray} from '@tensorflow/tfjs-core';

export type SimpleBinaryOperation = (a: number, b: number) => number;
export type SimpleBinaryKernelImpl =
    (aShape: number[], bShape: number[], aVals: TypedArray, bVals: TypedArray,
     dtype: DataType) => [TypedArray, number[]];
export type ComplexBinaryOperation =
    (aReal: number, aImag: number, bReal: number, bImag: number) => {
      real: number, imag: number
    };
export type ComplexBinaryKernelImpl =
    (aShape: number[], bShape: number[], aRealVals: Float32Array,
     aImagVals: Float32Array, bRealVals: Float32Array,
     bImagVals: Float32Array) => [TypedArray, TypedArray, number[]];
