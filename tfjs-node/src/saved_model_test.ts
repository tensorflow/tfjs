/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {loadSavedModel} from './saved_model';

describe('SavedModel', () => {
  it('load saved model and delete it', async () => {
    const model = await loadSavedModel(
        __dirname.slice(0, -3) + 'test_objects/times_two_int');
    await model.delete();
  });

  it('load wrong path', async done => {
    try {
      const model =
          await loadSavedModel(__dirname.slice(0, -3) + 'no_save_model');
      model.delete();
      done.fail();
    } catch (error) {
      expect(error.message)
          .toBe(
              'Faile to load SavedModel: Could not find SavedModel .pb ' +
              'or .pbtxt at supplied export directory path: ' +
              '/usr/local/google/home/kangyizhang/tensorflow/tfjs/' +
              'tfjs-node/no_save_model');
      done();
    }
  });

  it('delete a deleted saved model', async done => {
    try {
      const model = await loadSavedModel(
          __dirname.slice(0, -3) + 'test_objects/times_two_int');
      model.delete();
      // TODO(KANGYIZHANG): Replace this with model.predict() once it's
      // available.
      model.delete();
      done.fail();
    } catch (error) {
      expect(error.message).toBe('This SavedModel has been deleted.');
      done();
    }
  });
});
