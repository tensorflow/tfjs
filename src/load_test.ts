/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as frozen_model from './executor/frozen_model';
import * as frozen_model_json from './executor/frozen_model_json';
import * as tfc from './index';

const MODEL_URL_PB = 'https://test.com/tensorflowjs_model.pb';
const MODEL_URL_JSON = 'https://test.com/model.json';
const WEIGHT_MANIFEST_JSON = 'https://test.com/weights_manifest.json';
const TFHUB_MOBILENET =
    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2';
const ONPROGRESS_FUNC = (fraction: number) => {};

describe('external loading API', () => {
  describe('loadFrozenModel', () => {
    it('should support pb models', () => {
      spyOn(frozen_model, 'loadFrozenModel');

      tfc.loadFrozenModel(
          MODEL_URL_PB, WEIGHT_MANIFEST_JSON, {}, ONPROGRESS_FUNC);
      expect(frozen_model.loadFrozenModel)
          .toHaveBeenCalledWith(
              MODEL_URL_PB, WEIGHT_MANIFEST_JSON, {}, ONPROGRESS_FUNC);
    });
    it('should support pb models without manifest json', () => {
      spyOn(frozen_model, 'loadFrozenModel');

      tfc.loadFrozenModel(MODEL_URL_PB, undefined, {}, ONPROGRESS_FUNC);
      expect(frozen_model.loadFrozenModel)
          .toHaveBeenCalledWith(
              MODEL_URL_PB, WEIGHT_MANIFEST_JSON, {}, ONPROGRESS_FUNC);
    });
    it('should support json models', () => {
      spyOn(frozen_model_json, 'loadFrozenModel');

      tfc.loadFrozenModel(MODEL_URL_JSON, undefined, {}, ONPROGRESS_FUNC);
      expect(frozen_model_json.loadFrozenModel)
          .toHaveBeenCalledWith(MODEL_URL_JSON, {}, ONPROGRESS_FUNC);
    });
  });
  describe('loadGraphModel', () => {
    it('should support pb models', () => {
      spyOn(frozen_model, 'loadFrozenModel');

      tfc.loadGraphModel(
          MODEL_URL_PB, {requestInit: {}, onProgress: ONPROGRESS_FUNC});
      expect(frozen_model.loadFrozenModel)
          .toHaveBeenCalledWith(
              MODEL_URL_PB, WEIGHT_MANIFEST_JSON, {}, ONPROGRESS_FUNC);
    });
    it('should support json models', () => {
      spyOn(frozen_model_json, 'loadFrozenModel');

      tfc.loadGraphModel(
          MODEL_URL_JSON, {requestInit: {}, onProgress: ONPROGRESS_FUNC});
      expect(frozen_model_json.loadFrozenModel)
          .toHaveBeenCalledWith(MODEL_URL_JSON, {}, ONPROGRESS_FUNC);
    });
    it('should support tfhub models', () => {
      spyOn(frozen_model, 'loadTfHubModule');

      tfc.loadGraphModel(
          TFHUB_MOBILENET,
          {requestInit: {}, onProgress: ONPROGRESS_FUNC, fromTFHub: true});
      expect(frozen_model.loadTfHubModule)
          .toHaveBeenCalledWith(TFHUB_MOBILENET, {}, ONPROGRESS_FUNC);
    });
  });
});
