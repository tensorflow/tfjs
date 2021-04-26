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

import {Class as ProtoClass} from '../types/common';

/** Common options for all task library tasks. */
export interface CommonTaskLibraryOptions {
  /**
   * The number of threads to be used for TFLite ops that support
   * multi-threading when running inference with CPU. num_threads should be
   * greater than 0 or equal to -1. Setting num_threads to -1 has the effect to
   * let TFLite runtime set the value.
   *
   * Default to -1.
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
