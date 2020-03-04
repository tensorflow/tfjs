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
import {version_wasm} from './index';

describe('tfjs-core version consistency', () => {
  fit('dev-peer match', () => {
    const tfjsCoreDevDepVersion =
        packageJSON.devDependencies['@tensorflow/tfjs-core'];
    const tfjsCorePeerDepVersion =
        packageJSON.peerDependencies['@tensorflow/tfjs-core'];
    expect(tfjsCoreDevDepVersion).toEqual(tfjsCorePeerDepVersion);
  });

  fit('version.ts matches package version', () => {
    // tslint:disable-next-line:no-require-imports
    const expected = require('../package.json').version;
    expect(version_wasm).toBe(expected);
  });
});
