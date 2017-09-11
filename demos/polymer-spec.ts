/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
 * @fileoverview
 *
 * Defines an interface for creating Polymer elements in Typescript with the
 * correct typings. A Polymer element should be defined like this:
 *
 * ```
 * let MyElementPolymer = PolymerElement({
 *   is: 'my-polymer-element',
 *   properties: {
 *     foo: string,
 *     bar: Array
 *   }
 * });
 *
 * class MyElement extends MyElementPolymer {
 *   foo: string;
 *   bar: number[];
 *
 *   ready() {
 *     console.log('MyElement initialized!');
 *   }
 * }
 *
 * document.registerElement(MyElement.prototype.is, MyElement);
 * ```
 */

export type Spec = {
  is: string; properties: {
    [key: string]: (Function|{
      // tslint:disable-next-line:no-any
      type: Function, value?: any;
      reflectToAttribute?: boolean;
      readonly?: boolean;
      notify?: boolean;
      computed?: string;
      observer?: string;
    })
  };
  observers?: string[];
};

export function PolymerElement(spec: Spec) {
  // tslint:disable-next-line:no-any
  return Polymer.Class(spec as any) as {new (): PolymerHTMLElement};
}

export interface PolymerHTMLElement extends HTMLElement, polymer.Base {}
