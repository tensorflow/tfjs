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

import {upcastType} from './types';

describe('upcastType', () => {
  it('upcasts bool to bool', () => {
    expect(upcastType('bool', 'bool')).toBe('bool');
  });

  it('upcasts bool/int32 to int32', () => {
    expect(upcastType('bool', 'int32')).toBe('int32');
    expect(upcastType('int32', 'int32')).toBe('int32');
  });

  it('upcasts bool/int32/float32 to float32', () => {
    expect(upcastType('bool', 'float32')).toBe('float32');
    expect(upcastType('int32', 'float32')).toBe('float32');
    expect(upcastType('float32', 'float32')).toBe('float32');
  });

  it('upcasts bool/int32/float32/complex64 to complex64', () => {
    expect(upcastType('bool', 'complex64')).toBe('complex64');
    expect(upcastType('int32', 'complex64')).toBe('complex64');
    expect(upcastType('float32', 'complex64')).toBe('complex64');
    expect(upcastType('complex64', 'complex64')).toBe('complex64');
  });

  it('fails to upcast anything other than string with string', () => {
    expect(() => upcastType('bool', 'string')).toThrowError();
    expect(() => upcastType('int32', 'string')).toThrowError();
    expect(() => upcastType('float32', 'string')).toThrowError();
    expect(() => upcastType('complex64', 'string')).toThrowError();
    // Ok upcasting string to string.
    expect(upcastType('string', 'string')).toBe('string');
  });
});
