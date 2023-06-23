/**
 * @license
 * Copyright 2023 Google LLC.
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

import {matchAll} from './match_all_polyfill';

function format({match, index, input}: {match: string, index: number,
                                        input: string}) {
  const res: RegExpMatchArray = [match];
  res.index = index;
  res.input = input;
  return jasmine.objectContaining(res);
}

describe('matchAll', () => {
  it('finds all matches of a regexp on a string', () => {
    const input = 'asdfasdfasdfasdf';
    expect([...matchAll(input , /asd/g)]).toEqual([
      {match: 'asd', index: 0, input},
      {match: 'asd', index: 4, input},
      {match: 'asd', index: 8, input},
      {match: 'asd', index: 12, input},
    ].map(format));
  });

  it('supports regexp flags', () => {
    const input = 'asdfASDFasdfASDF';
    // Case sensitive
    expect([...matchAll(input, /asd/g)]).toEqual([
      {match: 'asd', index: 0, input},
      {match: 'asd', index: 8, input},
    ].map(format));

    // case insensitive
    expect([...matchAll('asdfASDFasdfASDF', /asd/gi)]).toEqual([
      {match: 'asd', index: 0, input},
      {match: 'ASD', index: 4, input},
      {match: 'asd', index: 8, input},
      {match: 'ASD', index: 12, input},
    ].map(format));
  });
});
