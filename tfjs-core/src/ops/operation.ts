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
import {ClosureCommand, ENGINE} from '../engine';
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

function buildOpAutoRecorder<T extends Function>(opFn: T, opName: string) {
  // tslint:disable-next-line:no-any
  return function opAutoRecorder(...args: any[]) {
    const {inputsCopy, tensors} = buildOpAutoRecorderInputs(args);
    return ClosureCommand.record(
        tensors.map(({tensor}) => tensor), (inputTensors: Tensor[]) => {
          for (let i = 0; i < inputTensors.length; ++i) {
            tensors[i].setter(inputTensors[i] as Tensor);
          }

          // The ClosureCommand executes the opFn in noRecordCommandScope to get
          // rid of commands from kernel execution.
          const outputTensors = opFn(...inputsCopy);

          if (outputTensors instanceof Tensor || Array.isArray(outputTensors)) {
            return outputTensors;
          }
          throw new Error(
              `Op auto recorder only supports Tensor and Tensor[] as outputs, got ${
                  outputTensors}`);
        }, {convertInputsToTensor: true, attrs: {opFn, opName}});
  };
}

type OpRecordingOptions = 'builtin'|'auto'|'none';

/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
export function op<T extends Function>(
    f: {[name: string]: T},
    recording: OpRecordingOptions|(() => OpRecordingOptions) = 'builtin'): T {
  // TODO(record-replay): For benchmarking record-replay with MobileNetV3, set
  // the default recording to 'builtin'.

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
  const rawOpFn = (...args: any[]) => {
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

  const getOpRecorder = (option: OpRecordingOptions) => {
    switch (option) {
      case 'none':
        return () => {
          throw new Error(`Op ${opName} does not support recording`);
        };
      case 'builtin':
        return rawOpFn;
      case 'auto':
        return buildOpAutoRecorder(rawOpFn, opName);
    }
  };

  const recordOpFn = (...args: unknown[]) => {
    const option = recording instanceof Function ? recording() : recording;

    console.log(`======= Op<${opName}> recording option: "${option}".`);
    return ENGINE.recordOpCommandScope(() => {
      return getOpRecorder(option)(...args);
    });
  };

  const opFn = (...args: unknown[]) => {
    // If the op is executed in a recordOpCommandScope, execute the op recorder
    // to push commands into the tape. Otherwise run the op in
    // noRecordCommandScope to avoid any
    // TODO: Remove recording scope and enforce recording through recordOpFn.
    if (ENGINE.state.activeCommandTape != null) {
      const option = recording instanceof Function ? recording() : recording;
      console.log(`====== Op<${opName}, option: "${
          option}"> is recorded by executing in a recordOpCommandScope.`);
      return getOpRecorder(option)(...args);
    }

    return ENGINE.noRecordCommandScope(() => rawOpFn(...args));
  };

  Object.defineProperty(opFn, 'name', {value: opName, configurable: true});
  Object.defineProperty(
      opFn, '_record', {value: recordOpFn, configurable: false});
  return opFn as unknown as T;
}
