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

import {DataType, stack, Tensor, tensor} from '@tensorflow/tfjs-core';
import {concat} from '@tensorflow/tfjs-core';
import {unstack} from '@tensorflow/tfjs-core';
import {tidy} from '@tensorflow/tfjs-core';
import {slice} from '@tensorflow/tfjs-core';

import {assertShapesMatchAllowUndefinedSize} from './tensor_utils';

/**
 * TensorList stores a container of `tf.Tensor` objects, which are accessible
 * via tensors field.
 *
 * In order to get a copy of the underlying list, use the copy method:
 * ```
 *    TensorList b = a.copy();
 *    b.tensors().pushBack(t);  // This does not modify a.tensors().
 * ```
 *
 * Note that this is not a deep copy: the memory locations of the underlying
 * tensors will still point to the same locations of the corresponding tensors
 * in the original.
 */

export class TensorList {
  /**
   *
   * @param tensors list of tensors
   * @param elementShape shape of each tensor
   * @param elementDtype data type of each tensor
   * @param maxNumElements The maximum allowed size of `tensors`. Defaults to -1
   *   meaning that the size of `tensors` is unbounded.
   */
  constructor(
      public tensors: Tensor[], public elementShape: number[],
      public elementDtype: DataType, public maxNumElements = -1) {}

  private _typeName: string;

  get typeName() {
    return this._typeName;
  }

  // Get a new TensorList containing a copy of the underlying tensor container.
  copy(): TensorList {
    return new TensorList(
        [...this.tensors], this.elementShape, this.elementDtype);
  }

  size() {
    return this.tensors.length;
  }

  /**
   * return a tensor that stacks all element together.
   * @param elementShape shape of each tensor
   * @param elementDtype data type of each tensor
   * @param numElements the number of elements to stack
   */
  stack(elementShape: number[], elementDtype: DataType, numElements = -1):
      Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }
    if (numElements !== -1 && this.tensors.length !== numElements) {
      throw new Error(`Operation expected a list with ${
          numElements} elements but got a list with ${
          this.tensors.length} elements.`);
    }
    assertShapesMatchAllowUndefinedSize(
        elementShape, this.elementShape, 'TensorList shape mismatch: ');
    return tidy(() => {
      const reshapedTensors =
          this.tensors.map(tensor => tensor.reshape(elementShape));
      return stack(reshapedTensors, 0);
    });
  }

  popBack(elementShape: number[], elementDtype: DataType): Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }

    if (this.size() === 0) {
      throw new Error('Trying to pop from an empty list.');
    }

    const tensor = this.tensors.pop();
    assertShapesMatchAllowUndefinedSize(
        tensor.shape, elementShape, 'TensorList shape mismatch: ');
    return tensor.reshape(elementShape);
  }

  pushBack(tensor: Tensor) {
    if (tensor.dtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          tensor.dtype}, but list elements ${this.elementDtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        tensor.shape, this.elementShape, 'TensorList shape mismatch: ');

    if (this.maxNumElements === this.size()) {
      throw new Error(`Trying to push element into a full list.`);
    }
    this.tensors.push(tensor);
  }

  resize(size: number) {
    if (size < 0) {
      throw new Error(
          `TensorListResize expects size to be non-negative. Got: ${size}`);
    }

    if (this.maxNumElements !== -1 && size > this.maxNumElements) {
      throw new Error(`TensorListResize input size ${
          size} is greater maxNumElement ${this.maxNumElements}.`);
    }
    this.tensors.length = size;
  }

  /**
   * Retrieve the element at the provided index
   * @param elementShape shape of the tensor
   * @param elementDtype dtype of the tensor
   * @param elementIndex index of the tensor
   */
  getItem(elementIndex: number, elementShape: number[], elementDtype: DataType):
      Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }
    if (elementIndex < 0 || elementIndex > this.tensors.length) {
      throw new Error(`Trying to access element ${
          elementIndex} in a list with ${this.tensors.length} elements.`)
    }

    if (this.tensors[elementIndex] == null) {
      throw new Error(`element at index ${elementIndex} is null.`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.tensors[elementIndex].shape, elementShape,
        'TensorList shape mismatch: ');

    return this.tensors[elementIndex];
  }

  /**
   * Set the tensor at the index
   * @param elementIndex index of the tensor
   * @param tensor the tensor to be inserted into the list
   */
  setItem(elementIndex: number, tensor: Tensor) {
    if (tensor.dtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          tensor.dtype}, but list elements ${this.elementDtype}`);
    }

    if (elementIndex < 0 ||
        this.maxNumElements !== -1 && elementIndex >= this.maxNumElements) {
      throw new Error(`Trying to set element ${
          elementIndex} in a list with max ${this.maxNumElements} elements.`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, tensor.shape, 'TensorList shape mismatch: ');

    this.tensors[elementIndex] = tensor;
  }

  /**
   * Return selected values in the TensorList as a packed Tensor. All of
   * selected values must have been written and their shapes must all match.
   * @param indices
   * @param dtype
   * @param shape
   */
  gather(indices: number[], elementDtype: DataType, shape: number[]): Tensor {
    if (elementDtype !== this.elementDtype) {
      throw new Error(`Invalid data types; op elements ${
          elementDtype}, but list elements ${this.elementDtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, shape, 'TensorList shape mismatch: ');

    indices = indices.slice(0, this.size());

    if (indices.length === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    return tidy(() => {
      const tensors = indices.map(i => this.tensors[i].reshape(shape));
      return stack(tensors, 0);
    });
  }

  /**
   * Return the values in the TensorList as a concatenated Tensor.
   */
  concat(dtype: DataType, shape: number[]): Tensor {
    if (!!dtype && dtype !== this.elementDtype) {
      throw new Error(`TensorList dtype is ${
          this.elementDtype} but concat requested dtype ${dtype}`);
    }

    assertShapesMatchAllowUndefinedSize(
        this.elementShape, shape, 'TensorList shape mismatch: ');

    if (this.size() === 0) {
      return tensor([], [0].concat(this.elementShape));
    }

    return tidy(() => {
      const tensors = this.tensors.map(t => t.reshape(shape));
      return concat(tensors, 0);
    });
  }
}

/**
 * Creates a TensorList which, when stacked, has the value of tensor.
 * @param tensor
 * @param shape
 */
export function fromTensor(tensor: Tensor, shape: number[]) {
  const dtype = tensor.dtype;
  if (tensor.shape.length < 1) {
    throw new Error(
        `Tensor must be at least a vector, but saw shape: ${tensor.shape}`);
  }

  const outputShape = tensor.shape.slice(1);
  assertShapesMatchAllowUndefinedSize(
      outputShape, shape, 'TensorList shape mismatch: ');

  const tensorList: Tensor[] = [];
  for (let i = 0; i < tensor.shape[0]; ++i) {
    const tmp = tensor.slice(i, i + 1).reshape(outputShape);
    tensorList.push(tmp);
  }
  return new TensorList(tensorList, shape, dtype);
}

/**
 * return a TensorList of the given size with empty elements.
 * @param elementShape the shape of the future elements of the list
 * @param elementDtype the desired type of elements in the list
 * @param numElements the number of elements to reserve
 */
export function reserve(
    elementShape: number[], elementDtype: DataType, numElements: number) {
  return new TensorList([], elementShape, elementDtype, numElements);
}


/**
 * Scatter the values of a Tensor in specific indices of a TensorList.
 * @param indices nummber[] values in [0, max_value). If the
 *    TensorList is not dynamic, max_value=size().
 * @param tensor Tensor input tensor.
 */
export function scatter(
    tensor: Tensor, indices: number[], elementShape: number[],
    numElements: number): TensorList {
  if (indices.length !== tensor.shape[0]) {
    throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${
        indices.length} vs. ${tensor.shape[0]}`);
  }

  const maxIndex = Math.max(...indices);

  if (numElements !== -1 && maxIndex >= numElements) {
    throw new Error(
        `Max index must be < array size (${maxIndex}  vs. ${numElements})`);
  }

  const list = new TensorList([], elementShape, tensor.dtype, numElements);
  const tensors = unstack(tensor, 0);
  indices.forEach((value, index) => {
    list.setItem(value, tensors[index]);
  });
  return list;
}

/**
 * Split the values of a Tensor into the TensorArray.
 * @param length number[] with the lengths to use when splitting value along
 *    its first dimension.
 * @param tensor Tensor, the tensor to split.
 */
export function split(
    tensor: Tensor, length: number[], elementShape: number[]) {
  let totalLength = 0;
  const cumulativeLengths = length.map(len => {
    totalLength += len;
    return totalLength;
  });

  if (totalLength !== tensor.shape[0]) {
    throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
  }

  const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
  const tensors: Tensor[] = tidy(() => {
    const tensors = [];
    tensor = tensor.reshape([1, totalLength, elementPerRow]);
    for (let i = 0; i < length.length; ++i) {
      const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
      const indices = [0, previousLength, 0];
      const sizes = [1, length[i], elementPerRow];
      tensors[i] = slice(tensor, indices, sizes).reshape(elementShape);
    }
    return tensors;
  });

  const list = new TensorList([], elementShape, tensor.dtype, length.length);

  for (let i = 0; i < tensors.length; i++) {
    list.setItem(i, tensors[i]);
  }
  return list;
}
