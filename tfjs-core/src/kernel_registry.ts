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

import {DataType} from './types';

export type GradSaveFunc = (save: DataId[]) => void;

export type DataId = object;

export type Attribute = number|number[]|boolean|boolean[]|string|string[];

/** Specifies the code to run when executing a kernel. */
export type KernelFunc = (params: {
  inputs: NamedDataMap,
  storage: {},
  attrs?: NamedAttrMap,
  save?: GradSaveFunc
}) => DataInfo|DataInfo[];

/** Holds metadata for a given tensor. */
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

/**
 * Returns the kernel function (code) associated with the provided names.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 */
export function getKernel(kernelName: string, backendName: string): KernelFunc {
  const key = makeKey(kernelName, backendName);
  return kernelRegistry[key];
}

export function getKernelRegistry(): {[key: string]: KernelFunc} {
  return kernelRegistry;
}

/**
 * Registers the function (forward pass) for the kernel in a global registry.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 * @param forward The function to run during the forward pass of the kernel.
 */
export function registerKernel(
    kernelName: string, backendName: string, kernelFunc: KernelFunc) {
  const key = makeKey(kernelName, backendName);
  if (key in kernelRegistry) {
    throw new Error(
        `The kernel '${kernelName}' for backend ` +
        `'${backendName}' is already registered`);
  }
  kernelRegistry[key] = kernelFunc;
}

/** Removes the function (forward pass) for the kernel from the registry. */
export function unregisterKernel(
    kernelName: string, backendName: string): void {
  const key = makeKey(kernelName, backendName);
  if (!(key in kernelRegistry)) {
    throw new Error(
        `The kernel '${kernelName}' for backend ` +
        `'${backendName}' is not registered`);
  }
  delete kernelRegistry[key];
}

const kernelRegistry: {[key: string]: KernelFunc} = {};

function makeKey(kernelName: string, backendName: string) {
  return `${backendName}_${kernelName}`;
}
