/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import * as tfwebClient from './tfweb_client';

describe('TFWebClient', () => {
  beforeEach(() => {});

  // The tbweb client is loaded through deps/tfweb_client.js, and we define its
  // type in src/tfweb_client.d.ts. This test makes sure that all the types
  // defined in tfweb_client.d.ts actually exist in the loaded JS client.
  it('should all its exported types defined', () => {
    // In test environment, tfweb namespace is in tfwebClient.default.
    //
    // TODO: figure out how to remove this workaround by updating the karma
    // configs.
    const tfweb = (tfwebClient as any).default.tfweb;
    for (const field of Object.keys(tfweb)) {
      expect(tfweb[field]).toBeDefined();
    }
    expect(tfweb['tfweb']['setWasmPath']).toBeDefined();
  });
});
