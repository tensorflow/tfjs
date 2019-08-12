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
import {ALL_ENVS, describeWithFlags} from '../jasmine_util';
import {op} from './operation';

describeWithFlags('operation', ALL_ENVS, () => {
  it('executes and preserves function name', () => {
    const f = () => 2;
    const opfn = op({'opName': f});

    expect(opfn.name).toBe('opName');
    expect(opfn()).toBe(2);
  });

  it('executes, preserves function name, strips underscore', () => {
    const f = () => 2;
    const opfn = op({'opName_': f});

    expect(opfn.name).toBe('opName');
    expect(opfn()).toBe(2);
  });

  it('throws when passing an object with multiple keys', () => {
    const f = () => 2;
    expect(() => op({'opName_': f, 'opName2_': f}))
        .toThrowError(/Please provide an object with a single key/);
  });
});
