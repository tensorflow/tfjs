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

const kernelRegistry: Map<string, KernelConfig> = new Map();

export type DataId = object;

/** These are extra non-tensor/primitive params passed to kernel functions. */
export type Attribute = number|number[]|boolean|boolean[]|string|string[];

/** Specifies the code to run when executing a kernel. */
export type KernelFunc = (params: {
  inputs: NamedTensorInfoMap,
  backend: {},
  attrs?: NamedAttrMap,
}) => TensorInfo|TensorInfo[];

/** Function that gets called after the backend initializes. */
export type KernelSetupFunc = (backend: {}) => void;
/** Function that gets called right before the backend is disposed. */
export type KernelDisposeFunc = KernelSetupFunc;

/** Config object for registering a kernel in the global registry. */
export interface KernelConfig {
  kernelName: string;
  backendName: string;
  kernelFunc: KernelFunc;
  setupFunc?: KernelSetupFunc;
  disposeFunc?: KernelDisposeFunc;
}

/** Holds metadata for a given tensor. */
export interface TensorInfo {
  dataId: DataId;
  shape: number[];
  dtype: DataType;
}

export interface NamedTensorInfoMap {
  [name: string]: TensorInfo;
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
export function getKernel(
    kernelName: string, backendName: string): KernelConfig {
  const key = makeKey(kernelName, backendName);
  return kernelRegistry.get(key);
}

export function getKernelsForBackend(backendName: string): KernelConfig[] {
  const it = kernelRegistry.entries();
  const result: KernelConfig[] = [];

  while (true) {
    const {done, value} = it.next();
    if (done) {
      break;
    }
    const [key, config] = value;
    const [backend, ] = key.split('_');
    if (backend === backendName) {
      result.push(config);
    }
  }
  return result;
}

/**
 * Registers the function (forward pass) for the kernel in a global registry.
 *
 * @param config A config object with the following properties:
 * - `kernelName` The official name of the kernel.
 * - `backendName` The official name of the backend.
 * - `kernelFunc` The function to run during the forward pass of the kernel.
 * - `setupFunc` Optional. Gets called once, after the backend initializes.
 * - `disposeFunc` Optional. Gets called once, right before the backend is
 * disposed.
 */
export function registerKernel(config: KernelConfig) {
  const {kernelName, backendName} = config;
  const key = makeKey(kernelName, backendName);
  if (kernelRegistry.has(key)) {
    throw new Error(
        `The kernel '${kernelName}' for backend ` +
        `'${backendName}' is already registered`);
  }
  kernelRegistry.set(key, config);
}

/**
 * Removes the kernel function from the registry.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 *
 */
export function unregisterKernel(
    kernelName: string, backendName: string): void {
  const key = makeKey(kernelName, backendName);
  if (!kernelRegistry.has(key)) {
    throw new Error(
        `The kernel '${kernelName}' for backend ` +
        `'${backendName}' is not registered`);
  }
  kernelRegistry.delete(key);
}

function makeKey(kernelName: string, backendName: string) {
  return `${backendName}_${kernelName}`;
}
