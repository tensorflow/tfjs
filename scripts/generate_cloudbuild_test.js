// Copyright 2020 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

const generateCloudbuild = require('./generate_cloudbuild').generateCloudbuild;
const yaml = require('js-yaml');
const fs = require('fs');
const path = require('path');

// These tests only detect changes in the generated file, and may need to be
// updated if the project structure changes. To update them, run
// 'yarn update-cloudbuild-tests`.
// TODO(mattsoulanille): When Jasmine is updated to >=3.3.0, Use
// jasmine.withContext to show the above message if the tests fail.
describe('generateCloudbuild', () => {
  it('generates the correct cloudbuild file for tfjs-core', () => {
    const expectedCloudbuild = yaml.safeLoad(fs.readFileSync(
        path.join('scripts/cloudbuild_tfjs_core_expected.yml')));
    const cloudbuild = generateCloudbuild(['tfjs-core'], /* print */ false);
    expect(cloudbuild).toEqual(expectedCloudbuild);
  });

  it('generates the correct cloudbuild file for tfjs-node', () => {
    const expectedCloudbuild = yaml.safeLoad(fs.readFileSync(
        path.join('scripts/cloudbuild_tfjs_node_expected.yml')));
    const cloudbuild = generateCloudbuild(['tfjs-node'], /* print */ false);
    expect(cloudbuild).toEqual(expectedCloudbuild);
  });
});
