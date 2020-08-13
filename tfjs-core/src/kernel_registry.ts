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
import {env} from './environment';

import {getGlobal} from './global_util';
import {NamedGradientMap} from './tape';
import {Tensor} from './tensor';
import {DataType, RecursiveArray} from './types';

const kernelRegistry =
    getGlobal('kernelRegistry', () => new Map<string, KernelConfig>());
const gradRegistry =
    getGlobal('gradRegistry', () => new Map<string, GradConfig>());

export type DataId = object;

type AttributeValue =
    number|number[]|boolean|boolean[]|string|string[]|NamedAttrMap;

/** These are extra non-tensor/primitive params passed to kernel functions. */
export type Attribute = AttributeValue|RecursiveArray<AttributeValue>;

/** Specifies the code to run when executing a kernel. */
export type KernelFunc = (params: {
  inputs: NamedTensorInfoMap,
  backend: {},
  attrs?: NamedAttrMap,
}) => TensorInfo|TensorInfo[];

/** The function to run when computing a gradient during backprop. */
export type GradFunc =
    (dy: Tensor|Tensor[], saved: Tensor[], attrs: NamedAttrMap) =>
        NamedGradientMap;

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

/** Config object for registering a gradient in the global registry. */
export interface GradConfig {
  kernelName: string;
  inputsToSave?: string[];
  // When saveAllInputs is true, all inputs will be saved. Only use this flag
  // if inputs is an array of Tensors.
  saveAllInputs?: boolean;
  outputsToSave?: boolean[];
  gradFunc: GradFunc;
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

/**
 * Returns the registered gradient info associated with the provided kernel.
 * @param kernelName The official TF kernel name.
 */
export function getGradient(kernelName: string): GradConfig {
  return gradRegistry.get(kernelName);
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
    console.warn(
        `The kernel '${kernelName}' for backend ` +
        `'${backendName}' is already registered`);
  }
  kernelRegistry.set(key, config);
}

/**
 * Registers a gradient function for a given kernel in the global registry,
 * to be used during the back-propagation of that kernel.
 *
 * @param config An object with the following properties:
 * - `kernelName` The name of the kernel that the gradient function is for.
 * - `gradFunc` The function to run during back-propagation.
 */
export function registerGradient(config: GradConfig) {
  const {kernelName} = config;

  if (gradRegistry.has(kernelName)) {
    // TODO (yassogba) after 3.0 assess whether we need to keep this gated
    // to debug mode.
    if (env().getBool('DEBUG')) {
      console.warn(`Overriding the gradient for '${kernelName}'`);
    }
  }
  gradRegistry.set(kernelName, config);
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

/** Removes the registered gradient from the global registry. */
export function unregisterGradient(kernelName: string): void {
  if (!gradRegistry.has(kernelName)) {
    throw new Error(
        `The gradient '${kernelName}' for backend is not registered`);
  }
  gradRegistry.delete(kernelName);
}

function makeKey(kernelName: string, backendName: string) {
  return `${backendName}_${kernelName}`;
}
