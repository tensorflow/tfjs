/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {AttributeError, RuntimeError, ValueError} from './errors';

describe('Error classes', () => {
  // tslint:disable-next-line:variable-name
  for (const SomeClass of [AttributeError, RuntimeError, ValueError]) {
    it('pass instanceof tests.', () => {
      const msg = 'Some message';
      const e = new SomeClass(msg);
      expect(e.message).toEqual(msg);
      expect(e instanceof SomeClass).toBe(true);
    });
  }
});
