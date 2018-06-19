/**
 * @license
 * Copyright 2018 Google LLC.
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// tslint:disable-next-line:no-require-imports
const packageJSON = require('../package.json');
import {version_converter} from './index';

describe('tfjs-core version consistency', () => {
  it('dev-peer match', () => {
    const tfjsCoreDevDepVersion =
        packageJSON.devDependencies['@tensorflow/tfjs-core'];
    const tfjsCorePeerDepVersion =
        packageJSON.peerDependencies['@tensorflow/tfjs-core'];
    expect(tfjsCoreDevDepVersion).toEqual(tfjsCorePeerDepVersion);
  });

  it('version.ts matches package version', () => {
    // tslint:disable-next-line:no-require-imports
    const expected = require('../package.json').version;
    expect(version_converter).toBe(expected);
  });
});
