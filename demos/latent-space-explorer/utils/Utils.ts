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

import {Array1D, NDArray, NDArrayMathCPU, Scalar} from 'deeplearn';

const math = new NDArrayMathCPU();

export function interpolateLinear(
    embedding0: Array1D, embedding1: Array1D, ratio: Scalar) {
  return math.add(
      embedding0,
      math.scalarTimesArray(ratio, math.sub(embedding1, embedding0)));
}

export function linearCombination(
    startPoint: Array1D, direction: Array1D, factor: Scalar) {
  return math.add(startPoint, math.scalarTimesArray(factor, direction));
}

export function linearCombinationNormalized(
    startPoint: Array1D, direction: Array1D, factor: Scalar) {
  const mag = norm(startPoint);
  const result = math.add(startPoint, math.scalarTimesArray(factor, direction));
  const normalizedResult = math.scalarTimesArray(mag, unit(result));
  return normalizedResult;
}

export function direction(left: Array1D, right: Array1D) {
  return unit(math.sub(right, left));
}

export function norm(v: NDArray) {
  return math.sqrt(math.sum(math.multiply(v, v)));
}

export function unit(v: NDArray) {
  const n = Scalar.new(norm(v).get());
  if (n.get() === 0) {
    console.warn('Divide by zero');
  }
  return math.arrayDividedByScalar(v, n);
}
