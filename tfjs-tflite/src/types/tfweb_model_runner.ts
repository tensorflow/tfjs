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

/** TFWebModelRunner class type. */
export declare interface TFWebModelRunnerClass {
  /**
   * The factory function to create a TFWebModelRunner instance.
   *
   * @param modelPath The path to load the TFLite model from.
   * @param options Available options.
   */
  create(modelPath: string, options: TFWebModelRunnerOptions):
      Promise<TFWebModelRunner>;
}

/**
 * The main TFWebModelRunner class interface.
 *
 * It is a wrapper around TFLite Interpreter. See
 * https://www.tensorflow.org/lite/guide/inference for more info about related
 * concepts.
 */
export declare interface TFWebModelRunner {
  /** Gets model inputs. */
  getInputs(): TFWebModelRunnerTensorInfo[];

  /** Gets model outputs. */
  getOutputs(): TFWebModelRunnerTensorInfo[];

  /**
   * Run inference.
   *
   * @return Whether the inference is successful or not.
   */
  infer(): boolean;

  /** Cleans up. */
  cleanUp(): void;
}

/** Options for TFWebModelRunner. */
export declare interface TFWebModelRunnerOptions {
  /**
   * Number of threads to use when running inference.
   *
   * Set this to -1 to allow TFLite runtime to automatically pick threads count
   * based on current environment.
   */
  numThreads: number;
}

/** Types of TFLite tensor data. */
export type TFLiteDataType =
    'int8'|'uint8'|'bool'|'int16'|'int32'|'uint32'|'float32'|'float64';

/** Stores metadata for a TFLite tensor. */
export declare interface TFWebModelRunnerTensorInfo {
  /** The id of the tensor (generated by TFLite runtime). */
  id: number;

  /** TFLite data type. */
  dataType: TFLiteDataType;

  /** The name of the TFLite tensor. */
  name: string;

  /** The shape of the tensor in string form, e.g. "2,3,5". */
  shape: string;

  /** Gets the direct access to the underlying buffer. */
  data(): Int8Array|Uint8Array|Int16Array|Int32Array|Uint32Array|Float32Array
      |Float64Array;
}
