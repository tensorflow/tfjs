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

import * as tfliteWebApiClient from './tflite_web_api_client';

describe('TFLiteWebAPIClient', () => {
  beforeEach(() => {});

  // The tflite web API client is loaded through deps/tflite_web_api_client.js,
  // and we define its type in src/tflite_web_api_client.d.ts. This test makes
  // sure that all the types defined in tflite_web_api_client.d.ts actually
  // exist in the loaded JS client.
  it('should all its exported types defined', () => {
    const tfliteWeb = tfliteWebApiClient.tfweb;
    for (const field of Object.keys(tfliteWeb)) {
      // tslint:disable-next-line: no-any
      expect((tfliteWeb as any)[field]).toBeDefined();
    }
    expect(tfliteWeb['tflite_web_api']['setWasmPath']).toBeDefined();
    expect(tfliteWeb['tflite_web_api']['getWasmFeatures']).toBeDefined();
  });
});
