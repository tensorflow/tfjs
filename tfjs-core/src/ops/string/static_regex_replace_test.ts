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

import * as tf from '../../index';
import { DataTypeFor } from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';

describeWithFlags('staticRegexReplace', ALL_ENVS, () => {
  it('replaces the first instance of a string', async () => {
    const result = tf.string.staticRegexReplace(
      ['this', 'is', 'a', 'test test'], 'test', 'result', false);

    expect(await result.data<DataTypeFor<string>>())
      .toEqual(['this', 'is', 'a', 'result test']);
  });

  it('replaces a string globally by default', async () => {
    const result = tf.string.staticRegexReplace(
      ['this', 'is', 'a', 'test test'], 'test', 'result');

    expect(await result.data<DataTypeFor<string>>())
      .toEqual(['this', 'is', 'a', 'result result']);
  });

  it('matches using regex', async () => {
    const result = tf.string.staticRegexReplace(
      ['This     will  have normal    whitespace'], ' +', ' ');

    expect(await result.data<DataTypeFor<string>>())
      .toEqual(['This will have normal whitespace']);
  });
});
