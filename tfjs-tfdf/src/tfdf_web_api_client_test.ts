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

import * as tfdfWebApiClient from './tfdf_web_api_client';

describe('TFDFWebAPIClient', () => {
  beforeEach(() => {});

  // The tflite web API client is loaded through deps/tfdf_web_api_client.js,
  // and we define its type in src/tfdf_web_api_client.d.ts. This test makes
  // sure that all the types defined in tfdf_web_api_client.d.ts actually
  // exist in the loaded JS client.
  it('should all its exported types defined', async () => {
    const tfdfWeb = await tfdfWebApiClient.tfweb;
    const model = await tfdfWeb.loadModelFromUrl(
        'https://achoum.github.io/yggdrasil_decision_forests_js_example/model.zip');
    console.log(model);
    expect(tfdfWeb.loadModelFromUrl).toBeDefined();
  });
});
