// Copyright 2019 Google LLC. All Rights Reserved.
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

const {google} = require('googleapis');

module.exports.nightly = async data => {
  const cloudbuild = google.cloudbuild('v1');
  const auth = await google.auth.getClient(
      {scopes: ['https://www.googleapis.com/auth/cloud-platform']});
  google.options({auth});
  const resp = await cloudbuild.projects.triggers.run({
    'projectId': 'learnjs-174218',
    'triggerId': '7423c985-2fd2-40f3-abe7-94d4c353eed0',
    'resource': {'branchName': 'master'}
  });
  console.log(resp);
};
