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

const kms = require('@google-cloud/kms');
const request = require('request-promise-native');

const projectId = 'learnjs-174218';
const locationId = 'global';
const keyRingId = 'tfjs';
const cryptoKeyId = 'enc'
const ciphertext =
    'CiQAkwyoIW0LcnxymzotLwaH4udVTQFBEN4AEA5CA+a3+yflL2ASPQAD8BdZnGARf78MhH5T9rQqyz9HNODwVjVIj64CTkFlUCGrP1B2HX9LXHWHLmtKutEGTeFFX9XhuBzNExA=';
const browserStackUploadUrl =
    'https://api-cloud.browserstack.com/app-automate/upload';
const browserStackUser = 'deeplearnjs1';
const testAppUrl =
    'https://storage.googleapis.com/tfjs-rn/integration-tests/app-debug.apk';
const appUploadId = 'tfjs-rn-integration-android';

async function sync_reactnative(event, context, callback) {
  const client = new kms.KeyManagementServiceClient();
  const name =
      client.cryptoKeyPath(projectId, locationId, keyRingId, cryptoKeyId);

  const [result] = await client.decrypt({name, ciphertext});
  const browserStackKey = result.plaintext.toString();

  try {
    const syncRes = await request.post(browserStackUploadUrl, {
      auth: {
        user: browserStackUser,
        pass: browserStackKey,
      },
      form: {
        data: JSON.stringify({'url': testAppUrl, 'custom_id': appUploadId}),
      }
    });
    sendChatMsg(
        process.env.BOTS_HANGOUTS_URL,
        'Success syncing tfjs-react-native integration test app to BrowserStack');
  } catch (e) {
    console.log('Error syncing app to browserstack', e);
    sendChatMsg(
        process.env.HANGOUTS_URL,
        'Error syncing tfjs-react-native integration test app to BrowserStack');
  }
};

async function sendChatMsg(url, msg) {
  const res = await request(url, {
    resolveWithFullResponse: true,
    method: 'POST',
    json: true,
    body: {text: msg},
  });
}

module.exports.sync_reactnative = sync_reactnative;
