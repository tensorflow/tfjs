/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {getUid} from '../backend/state';

describe('getUID ', () => {
  it('second UID is different.', () => {
    const name = 'def';
    const firstUID = getUid(name);
    const secondUID = getUid(name);
    expect(secondUID).not.toEqual(firstUID);
  });

  it('with no prefix works and returns different UIDs.', () => {
    const firstUID = getUid();
    const secondUID = getUid();
    expect(firstUID).not.toEqual(secondUID);
  });
});
