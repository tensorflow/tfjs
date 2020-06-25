/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// Note that the identifier globalNameSpace is scoped to this module, but will
// always resolve to the same global object regardless of how the module is
// resolved.
// tslint:disable-next-line:no-any
let globalNameSpace: {_tfGlobals: Map<string, any>};
// tslint:disable-next-line:no-any
export function getGlobalNamespace(): {_tfGlobals: Map<string, any>} {
  if (globalNameSpace == null) {
    // tslint:disable-next-line:no-any
    let ns: any;
    if (typeof (window) !== 'undefined') {
      ns = window;
    } else if (typeof (global) !== 'undefined') {
      ns = global;
    } else if (typeof (process) !== 'undefined') {
      ns = process;
    } else if (typeof (self) !== 'undefined') {
      ns = self;
    } else {
      throw new Error('Could not find a global object');
    }
    globalNameSpace = ns;
  }
  return globalNameSpace;
}

// tslint:disable-next-line:no-any
function getGlobalMap(): Map<string, any> {
  const ns = getGlobalNamespace();
  if (ns._tfGlobals == null) {
    ns._tfGlobals = new Map();
  }
  return ns._tfGlobals;
}

/**
 * Returns a globally accessible 'singleton' object.
 *
 * @param key the name of the object
 * @param init a function to initialize to initialize this object
 *             the first time it is fetched.
 */
export function getGlobal<T>(key: string, init: () => T): T {
  const globalMap = getGlobalMap();
  if (globalMap.has(key)) {
    return globalMap.get(key);
  } else {
    const singleton = init();
    globalMap.set(key, singleton);
    return globalMap.get(key);
  }
}
