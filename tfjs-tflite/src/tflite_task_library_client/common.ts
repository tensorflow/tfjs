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

import * as tfliteWebAPIClient from '../tflite_web_api_client';
import {BaseTaskLibrary, Class as ProtoClass} from '../types/common';

/** Common options for all task library tasks. */
export interface CommonTaskLibraryOptions {
  /**
   * The number of threads to be used for TFLite ops that support
   * multi-threading when running inference with CPU. num_threads should be
   * greater than 0 or equal to -1. Setting num_threads to -1 has the effect to
   * let TFLite runtime set the value.
   *
   * Default to number of physical CPU cores, or -1 if WASM multi-threading is
   * not supported by user's browser.
   */
  numThreads?: number;
}

/** A single class in the classification result. */
export interface Class {
  /** The name of the class. */
  className: string;

  /** The probability/score of the class. */
  probability: number;
}

/** Convert proto Class array to our own Class array. */
export function convertProtoClassesToClasses(protoClasses: ProtoClass[]):
    Class[] {
  const classes: Class[] = [];
  protoClasses.forEach(cls => {
    classes.push({
      className: cls.getDisplayName() || cls.getClassName(),
      probability: cls.getScore(),
    });
  });
  return classes;
}

/** The global function to set WASM path. */
export const setWasmPath = tfliteWebAPIClient.tfweb.tflite_web_api.setWasmPath;

/** The global function to get supported WASM features */
export const getWasmFeatures =
    tfliteWebAPIClient.tfweb.tflite_web_api.getWasmFeatures;

/** The base class for all task library clients. */
export class BaseTaskLibraryClient {
  constructor(protected instance: BaseTaskLibrary) {}

  cleanUp() {
    this.instance.cleanUp();
  }
}

/** Gets the number of threads for best performance. */
export async function getDefaultNumThreads(): Promise<number> {
  const supportMultiThreading =
      (await tfliteWebAPIClient.tfweb.tflite_web_api.getWasmFeatures())
          .multiThreading;
  return supportMultiThreading ? navigator.hardwareConcurrency / 2 : -1;
}
