/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {tidy} from './backends/tracking';

/**
 * Decorator for wrapping functions that perform math operations on
 * NDArrays. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
export function operation(
    target: {}, name: string, descriptor: PropertyDescriptor) {
  const fn = descriptor.value;
  // tslint:disable-next-line:no-any
  descriptor.value = (...args: any[]) => {
    return tidy(name, () => fn(...args));
  };
  return descriptor;
}
