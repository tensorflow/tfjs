/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// tslint:disable-next-line:no-require-imports
import {version_layers} from './index';

describe('tfjs-core version consistency', () => {
  it('version.ts matches package version', () => {
    // tslint:disable-next-line:no-require-imports
    const expected = require('../package.json').version;
    expect(version_layers).toBe(expected);
  });
});
