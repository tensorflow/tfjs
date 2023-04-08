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
import {ClosureCommand, CommandTape, ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {isPromise} from '../util';

export const OP_SCOPE_SUFFIX = '__op';

function buildOpAutoRecorderInputs(inputs: unknown[]) {
  const tensors: {tensor: Tensor, setter: (tensor: Tensor) => void}[] = [];

  const extract =
      (x: unknown, setter: (tensor: Tensor) => void) => {
        if (x instanceof Tensor) {
          tensors.push({tensor: x, setter});
          return x;
        }
        if (Array.isArray(x)) {
          const arrayCopy: unknown[] = [];
          for (let i = 0; i < x.length; ++i) {
            arrayCopy[i] = extract(x[i], (t) => arrayCopy[i] = t);
          }
          return arrayCopy;
        }
        if (typeof x === 'object') {
          // The order is consistent and determined.
          const objectCopy: Record<string, unknown> = {};
          for (const [key, val] of Object.entries(x)) {
            objectCopy[key] = extract(val, (t) => objectCopy[key] = t);
          }
          return objectCopy
        }

        // Should be primitive type, returns original reference.
        return x;
      }

  const inputsCopy = extract(inputs, () => {}) as unknown[];
  return {inputsCopy, tensors};
}

function buildOpAutoRecorder<T extends Function>(opFn: T) {
  // tslint:disable-next-line:no-any
  return function opAutoRecorder(...args: any[]) {
    const {inputsCopy, tensors} = buildOpAutoRecorderInputs(args);
    return ClosureCommand.record<
        Tensor>(tensors.map(({tensor}) => tensor), (inputTensors: Tensor[]) => {
      for (let i = 0; i < inputTensors.length; ++i) {
        tensors[i].setter(inputTensors[i] as Tensor);
      }

      const outputTensors =
          ENGINE.state.activeCommandTape.noCommandScope(() => {
            // Execute the opFn in noCommandScope to get rid of commands from
            // kernel execution, since op auto recorder produces one single
            // command to include all computations in the op.
            return opFn(...inputsCopy);
          });

      if (outputTensors instanceof Tensor) {
        return outputTensors;
      }
      if (Array.isArray(outputTensors)) {
        return outputTensors;
      }
      throw new Error(
          `Op auto recorder only supports Tensor and Tensor[] as outputs, got ${
              outputTensors}`);
    });
  };
}

/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
export function op<T extends Function>(
    f: {[name: string]: T}, record: 'builtin'|'auto'|'none' = 'auto'): T {
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

  let recorder: (...args: unknown[]) => unknown;
  switch (record) {
    case 'none':
      recorder = () => {
        throw new Error(`Op ${opName} does not support recording`);
      };
      break;
    case 'builtin':
      recorder = f2;
      break;
    case 'auto':
      recorder = buildOpAutoRecorder(f2);
      break;
  }

  const recordFn = (...args: unknown[]) => {
    ENGINE.state.activeCommandTape = new CommandTape();
    const outputs = recorder(...args);

    const result = {
      tape: ENGINE.state.activeCommandTape,
      outputs: outputs,
    };
    ENGINE.state.activeCommandTape = undefined;
    return result;
  };

  Object.defineProperty(f2, 'name', {value: opName, configurable: true});
  Object.defineProperty(f2, 'record', {value: recordFn, configurable: false});
  return f2 as unknown as T;
}
