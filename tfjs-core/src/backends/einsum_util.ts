/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

/**
 * Utility functions for computing einsum (tensor contraction and summation
 * based on Einstein summation.)
 */

import {Tensor} from '../tensor';
import {assert} from '../util_base';

const ARROW = '->';
const ARROW_REGEX = /->/g;
const COMMA = ',';
const ELLIPSIS = '...';

/**
 * Parse an equation for einsum.
 *
 * @param equation The einsum equation (e.g., "ij,jk->ik").
 * @param numTensors Number of tensors provided along with `equation`. Used to
 *   check matching number of input tensors.
 * @returns An object consisting of the following fields:
 *   - allDims: all dimension names as strings.
 *   - summedDims: a list of all dimensions being summed over, as indices to
 *     the elements of `allDims`.
 *   - idDims: indices of the dimensions in each input tensor, as indices to
 *     the elements of `allDims.
 */
export function decodeEinsumEquation(equation: string, numTensors: number): {
  allDims: string[],
  summedDims: number[],
  idDims: number[][],
} {
  equation = equation.replace(/\s/g, '');  // Remove witespace in equation.
  const numArrows =
      (equation.length - equation.replace(ARROW_REGEX, '').length) /
      ARROW.length;
  if (numArrows < 1) {
    throw new Error('Equations without an arrow are not supported.');
  } else if (numArrows > 1) {
    throw new Error(`Equation must contain exactly one arrow ("${ARROW}").`);
  }
  const [inputString, outputString] = equation.split(ARROW);
  assert(
      inputString.indexOf(ELLIPSIS) === -1,
      () => `The ellipsis notation ("${ELLIPSIS}") is not supported yet.`);
  const inputTerms = inputString.split(COMMA);
  const numInputs = inputTerms.length;
  if (numTensors !== numInputs) {
    throw new Error(
        `Expected ${numInputs} input tensors, received ${numTensors}`);
  }
  if (numInputs > 2) {
    throw new Error(
        'Support for more than 2 input tensors is not implemented yet.');
  }

  const allDims: string[] = [];
  for (let i = 0; i < outputString.length; ++i) {
    const dimName = outputString[i];
    if (!inputTerms.some(inputTerm => inputTerm.indexOf(dimName) !== -1)) {
      throw new Error(
          `Output subscripts contain the label ${dimName} ` +
          `not present in the input subscripts.`);
    }
    if (allDims.indexOf(dimName) === -1) {
      allDims.push(dimName);
    }
  }
  for (let i = 0; i < inputString.length; ++i) {
    const dimName = inputString[i];
    if (allDims.indexOf(dimName) === -1 && dimName !== COMMA) {
      allDims.push(dimName);
    }
  }

  const idDims: number[][] = new Array<number[]>(inputTerms.length);
  for (let i = 0; i < numInputs; ++i) {
    if (new Set(inputTerms[i].split('')).size !== inputTerms[i].length) {
      throw new Error(
          `Found duplicate axes in input component ${inputTerms[i]}. ` +
          `Support for duplicate axes in input is not implemented yet.`);
    }
    idDims[i] = [];
    for (let j = 0; j < inputTerms[i].length; ++j) {
      idDims[i].push(allDims.indexOf(inputTerms[i][j]));
    }
  }

  const numDims = allDims.length;          // Number of unique dimensions.
  const numOutDims = outputString.length;  // Number of output dimensions.
  const summedDims: number[] = [];         // Dimensions being summed over.
  for (let i = numOutDims; i < numDims; ++i) {
    summedDims.push(i);
  }
  return {allDims, summedDims, idDims};
}

/**
 * Get the permutation for a given input tensor.
 *
 * @param nDims Total number of dimension of all tensors involved in the einsum
 *   operation.
 * @param idDims Dimension indices involve in the tensor in question.
 * @returns An object consisting of the following fields:
 *   - permutationIndices: Indices to permute the axes of the tensor with.
 *   - expandDims: Indices to the dimension that need to be expanded from the
 *     tensor after permutation.
 */
export function getEinsumPermutation(nDims: number, idDims: number[]):
    {permutationIndices: number[], expandDims: number[]} {
  let permutationIndices: number[] = new Array<number>(nDims);
  permutationIndices.fill(-1);
  for (let i = 0; i < idDims.length; ++i) {
    permutationIndices[idDims[i]] = i;
  }
  const expandDims: number[] = [];
  for (let i = 0; i < nDims; ++i) {
    if (permutationIndices[i] === -1) {
      expandDims.push(i);
    }
  }
  permutationIndices = permutationIndices.filter(d => d !== -1);
  return {permutationIndices, expandDims};
}

/**
 * Checks that the dimension sizes from different input tensors match the
 * equation.
 */
export function checkEinsumDimSizes(
    nDims: number, idDims: number[][], tensors: Tensor[]) {
  const dimSizes: number[] = new Array<number>(nDims);
  for (let i = 0; i < tensors.length; ++i) {
    const shape: number[] = tensors[i].shape;
    for (let j = 0; j < idDims[i].length; ++j) {
      if (dimSizes[idDims[i][j]] === undefined) {
        dimSizes[idDims[i][j]] = shape[j];
      } else {
        assert(
            dimSizes[idDims[i][j]] === shape[j],
            () => `Expected dimension ${dimSizes[idDims[i][j]]} at axis ${j} ` +
                `of input shaped ${JSON.stringify(shape)}, ` +
                `but got dimension ${shape[j]}`);
      }
    }
  }
}

/**
 * Gets path of computation for einsum.
 *
 * @param summedDims indices to the dimensions being summed over.
 * @param idDims A look up table for the dimensions present in each input
 *     tensor. Each consituent array contains indices for the dimensions in the
 *     corresponding input tensor.
 *
 * @return A map with two fields:
 *   - path: The path of computation, with each element indicating the dimension
 *     being summed over after the element-wise multiplication in that step.
 *   - steps: With the same length as `path`. Each element contains the indices
 *     to the input tensors being used for element-wise multiplication in the
 *     corresponding step.
 */
export function getEinsumComputePath(summedDims: number[], idDims: number[][]):
    {path: number[], steps: number[][]} {
  const path: number[] = summedDims;
  const steps: number[][] = [];
  let nSteps = 0;
  if (summedDims.length === 0) {
    // Einsum that involes no summing: e.g., transpose and outer product.
    path.push(-1);
  }
  nSteps = summedDims.length + 1;
  for (let i = 0; i < nSteps; ++i) {
    steps.push([]);
  }
  const computedTermIndices: number[] = [];
  for (let i = 0; i < path.length; ++i) {
    const summedDim = path[i];
    const termIndices = findTermsWithDim(idDims, summedDim);
    for (const termIndex of termIndices) {
      if (computedTermIndices.indexOf(termIndex) === -1) {
        steps[i].push(termIndex);
        computedTermIndices.push(termIndex);
      }
    }
  }
  return {path, steps};
}

/** Determines if an axes permutation is the identity permutation. */
export function isIdentityPermutation(perm: number[]): boolean {
  return perm.every((dim: number, index: number) => dim === index);
}

function findTermsWithDim(idDims: number[][], dim: number): number[] {
  const termIndices: number[] = [];
  for (let i = 0; i < idDims.length; ++i) {
    if (idDims[i].length === 0 || idDims[i].indexOf(dim) !== -1 || dim === -1) {
      termIndices.push(i);
    }
  }
  return termIndices;
}
