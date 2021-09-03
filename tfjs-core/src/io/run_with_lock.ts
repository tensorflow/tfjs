import { isPromise } from "../util_base";
import { purgeLocalStorageArtifacts } from "./local_storage";

/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

let lock = Promise.resolve();

export function runWithLock(f: (done?: DoneFn) => Promise<void> | void) {
  return () => {
    lock = lock.then(async () => {
      let done: DoneFn;
      const donePromise = new Promise<void>((resolve, reject) => {
        done = (() => {
          resolve();
        }) as DoneFn;
        done.fail = (message?) => {
          reject(message)
        }
      });

      purgeLocalStorageArtifacts();
      const result = f(done);

      if (isPromise(result)) {
        await result;
      }
      else {
        await donePromise;
      }
    });
    return lock;
  }
}
