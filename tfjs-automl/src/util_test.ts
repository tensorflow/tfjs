/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import {util} from '@tensorflow/tfjs-core';
import {loadDictionary} from './util';

describe('load dictionary', () => {
  it('relative url to model.json', async () => {
    spyOn(util, 'fetch').and.callFake((dictUrl: string) => {
      expect(dictUrl).toBe('dict.txt');
      return {text: async () => 'first\nsecond\nthird'};
    });
    const res = await loadDictionary('model.json');
    expect(res).toEqual(['first', 'second', 'third']);
  });

  it('relative url to model.json with a base path', async () => {
    spyOn(util, 'fetch').and.callFake((dictUrl: string) => {
      expect(dictUrl).toBe('base/path/dict.txt');
      return {text: async () => 'first\nsecond\nthird'};
    });
    const res = await loadDictionary('base/path/model.json');
    expect(res).toEqual(['first', 'second', 'third']);
  });

  it('absolute url to model.json', async () => {
    spyOn(util, 'fetch').and.callFake((dictUrl: string) => {
      expect(dictUrl).toBe('/dict.txt');
      return {text: async () => 'first\nsecond\nthird\n'};
    });
    const res = await loadDictionary('/model.json');
    expect(res).toEqual(['first', 'second', 'third']);
  });

  it('absolute url to model.json with a base path', async () => {
    spyOn(util, 'fetch').and.callFake((dictUrl: string) => {
      expect(dictUrl).toBe('/base/path/dict.txt');
      return {text: async () => 'first\nsecond\nthird\n'};
    });
    const res = await loadDictionary('/base/path/model.json');
    expect(res).toEqual(['first', 'second', 'third']);
  });
});
