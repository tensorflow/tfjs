/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */

import {TensorContainer, TensorContainerArray, TensorContainerObject} from '@tensorflow/tfjs-core/dist/tensor_types';

import {Dataset} from '.';
import {LazyIterator} from './iterators/lazy_iterator';

/**
 * JSON-like type representing a nested structure of primitives or Tensors.
 */
export type DataElement = TensorContainer;

export type DataElementObject = TensorContainerObject;

export type DataElementArray = TensorContainerArray;

export type PrimitiveOrT<T> = void|string|number|boolean|T;

// Maybe this should be called 'NestedContainer'-- that's just a bit unwieldy.
export type Container<T> = ContainerObject<T>|ContainerArray<T>;

export type ContainerOrT<T> = Container<T>|T;

export interface ContainerObject<T> {
  [x: string]: ContainerOrT<T>;
}
export interface ContainerArray<T> extends Array<ContainerOrT<T>> {}

/**
 * A nested structure of Datasets, used as the input to zip().
 */
export type DatasetContainer = Container<Dataset<DataElement>>;

/**
 * A nested structure of LazyIterators, used as the input to zip().
 */
export type IteratorContainer = Container<LazyIterator<DataElement>>;

/**
 * Types supported by FileChunkIterator in both Browser and Node Environment.
 */
export type FileElement = File|Blob|Uint8Array;
