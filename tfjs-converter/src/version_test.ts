/**
 * @license
 * Copyright 2018 Google LLC.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {version_converter} from './index';

describe('tfjs-core version consistency', () => {
  it('dev-peer match', () => {
    const tfjsCoreDevDepVersion =
        // tslint:disable-next-line:no-require-imports
        require('tfjs-converter/package.json').devDependencies['@tensorflow/tfjs-core'];

    const tfjsCorePeerDepVersion =
        // tslint:disable-next-line:no-require-imports
        require('tfjs-converter/package.json').peerDependencies['@tensorflow/tfjs-core'];
    expect(tfjsCoreDevDepVersion).toEqual(tfjsCorePeerDepVersion);
  });

  it('version.ts matches package version', () => {
    // tslint:disable-next-line:no-require-imports
    const expected = require('tfjs-converter/package.json').version;
    expect(version_converter).toBe(expected);
  });
});
