#!/usr/bin/env node

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

/**
 * Normalized kernels names to the variable name used in code for the kernel
 * config.
 */
export function kernelNameToVariableName(kernelName: string) {
  return kernelName.charAt(0).toLowerCase() + kernelName.slice(1);
}

/**
 * Given an op name returns the name of the file that would export that op.
 */
export function opNameToFileName(opName: string) {
  // add exceptions here.
  if (opName === 'isNaN') {
    return 'is_nan';
  }
  return opName.replace(/[A-Z]/g, (s: string) => `_${s.toLowerCase()}`);
}
