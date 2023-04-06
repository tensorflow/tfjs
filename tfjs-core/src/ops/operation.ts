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
 * =============================================================================
 */
import {ClosureCommand, ENGINE, TensorPlaceholder} from '../engine';
import {Tensor} from '../tensor';
import {isPromise} from '../util';

export const OP_SCOPE_SUFFIX = '__op';

// tslint:disable-next-line:no-any
function buildOpRecorderInputs(inputs: any[]) {
  const tensors: {tensor: Tensor, setter: (tensor: Tensor) => void}[] = [];

  const extract =
      (vals: any, setter: (tensor: Tensor) => void) => {
        if (vals instanceof Tensor) {
          tensors.push({tensor: vals, setter});
          return vals;
        }
        if (Array.isArray(vals)) {
          const clonedArray: any[] = [];
          for (let i = 0; i < vals.length; ++i) {
            clonedArray[i] = extract(vals[i], (t) => clonedArray[i] = t);
          }
          return clonedArray;
        }
        if (typeof vals === 'object') {
          // The order is consistent and determined.
          const clonedObject: Record<string, any> = {};
          for (const [key, val] of Object.entries(vals)) {
            clonedObject[key] = extract(val, (t) => clonedObject[key] = t);
          }
          return clonedObject
        }

        return vals;
      }

  const clonedInputs = extract(inputs, () => {});
  return {clonedInputs, tensors};
}

function buildAutoOpRecorder<T extends Function>(opFn: T) {
  // tslint:disable-next-line:no-any
  return function opRecorder(...args: any[]) {
    const {clonedInputs, tensors} = buildOpRecorderInputs(args);
    ENGINE.state.activateCommandTape.pushAndExecute(
        ClosureCommand, tensors.map(({tensor}) => tensor), (inputTensors) => {
          for (let i = 0; i < inputTensors.length; ++i) {
            tensors[i].setter(inputTensors[i] as Tensor);
          }
          const outputTensors = opFn(clonedInputs);
          if (Array.isArray(outputTensors)) {
            return outputTensors;
          } if ()
        });
  };
}

/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
export function op<T extends Function>(f: {[name: string]: T}): T {
  const keys = Object.keys(f);
  if (keys.length !== 1) {
    throw new Error(
        `Please provide an object with a single key ` +
        `(operation name) mapping to a function. Got an object with ` +
        `${keys.length} keys.`);
  }

  let opName = keys[0];
  const fn = f[opName];

  // Strip the underscore from the end of the function name.
  if (opName.endsWith('_')) {
    opName = opName.substring(0, opName.length - 1);
  }

  // add an __op suffix to distinguish ops from kernels in tf.profile
  opName = opName + OP_SCOPE_SUFFIX;

  // tslint:disable-next-line:no-any
  const f2 = (...args: any[]) => {
    ENGINE.startScope(opName);
    try {
      const result = fn(...args);
      if (isPromise(result)) {
        console.error('Cannot return a Promise inside of tidy.');
      }
      ENGINE.endScope(result);
      return result;
    } catch (ex) {
      ENGINE.endScope(null);
      throw ex;
    }
  };
  Object.defineProperty(f2, 'name', {value: opName, configurable: true});

  // tslint:disable-next-line:no-any
  return f2 as any as T;
}
