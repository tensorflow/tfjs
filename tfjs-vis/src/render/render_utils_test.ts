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

import {visor} from '../index';
import {getDrawArea, shallowEquals} from './render_utils';

describe('shallowEqual', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('returns true for similar objects', async () => {
    const a = {
      stringProp: 'astring',
      numProp: 55,
      boolProp: true,
    };

    const b = {
      stringProp: 'astring',
      boolProp: true,
      numProp: 55,
    };

    expect(shallowEquals(a, b)).toBe(true);
  });

  it('returns false for different objects', async () => {
    const a = {
      stringProp: 'astring',
      numProp: 55,
      boolProp: false,
    };

    const b = {
      stringProp: 'astring',
      numProp: 55,
      boolProp: true,
    };

    expect(shallowEquals(a, b)).toBe(false);
  });

  it('returns true for similar objects (array ref)', async () => {
    // tslint:disable-next-line:no-any
    const ref: any[] = [];

    const a = {
      stringProp: 'astring',
      numProp: 55,
      refProp: ref,
    };

    const b = {
      numProp: 55,
      stringProp: 'astring',
      refProp: ref,
    };

    expect(shallowEquals(a, b)).toBe(true);
  });
});

describe('getDrawArea', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="container"></div>';
  });

  it('works with HTMLElement', async () => {
    const el = document.getElementById('container') as HTMLElement;
    expect(getDrawArea(el)).toEqual(el);
  });

  it('works with a surface', async () => {
    const surface = visor().surface({name: 'test'});
    expect(getDrawArea(surface)).toEqual(surface.drawArea);
  });

  it('fails with other stuff', async () => {
    const surface = visor().surface({name: 'test'});
    //@ts-ignore
    expect(() => getDrawArea('not-a-surface')).toThrow();
  });
});
