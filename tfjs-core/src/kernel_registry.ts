/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {GradSaveFunc} from './tensor_types';
import {DataType} from './types';

interface Kernel {
  func: ForwardFunc;
}

export const kernelRegistry: {[name: string]: Kernel} = {};

export type GradSaveFunc = (save: DataId[]) => void;

export type DataId = object;

export type Attribute = number|number[]|boolean|boolean[]|string|string[];

export type ForwardFunc = (params: {
  inputs: NamedDataMap,
  storage: {},
  attrs?: NamedAttrMap,
  save?: GradSaveFunc
}) => DataInfo|DataInfo[];

export type BackwardFunc =
    (dy: DataId, saved: DataId[], attrs: NamedAttrMap, storage: Storage) =>
        NamedDataMap;

export interface DataInfo {
  dataId: DataId;
  shape: number[];
  dtype: DataType;
}

export interface NamedDataMap {
  [name: string]: DataInfo;
}

export interface NamedAttrMap {
  [name: string]: Attribute;
}

export function registerKernel(
    kernelName: string, backendName: string, func: ForwardFunc) {
  const key = `${backendName}_${kernelName}`;
  if (key in kernelRegistry) {
    throw new Error(
        `Kernel ${kernelName} for backend ` +
        `${backendName} is already registered`);
  }
  kernelRegistry[key] = {func};
}

export function registerGradient(
    kernelName: string, backendName: string, func: BackwardFunc) {}
