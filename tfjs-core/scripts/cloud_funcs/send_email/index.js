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

const Mailgun = require('mailgun-js');
const humanizeDuration = require('humanize-duration');
const request = require('request-promise-native');
const config = require('./config.json');
const fetch = require('node-fetch');

const mailgun = new Mailgun({
  apiKey: process.env.MAILGUN_API_KEY,
  domain: config.MAILGUN_DOMAIN,
});

const TRIGGER_ID = '7423c985-2fd2-40f3-abe7-94d4c353eed0';

// The main function called by Cloud Functions.
module.exports.send_email = async event => {
  // Parse the build information.
  const build = JSON.parse(new Buffer(event.data, 'base64').toString());
  // FYI: you can add 'SUCCESS' to monitor successful builds also.
  const status = ['FAILURE', 'INTERNAL_ERROR', 'TIMEOUT', 'CANCELLED'];
  // Email only when the build has failed.
  if (status.indexOf(build.status) === -1) {
    return;
  }
  // Email only on nightly builds.
  if (build.buildTriggerId !== TRIGGER_ID) {
    return;
  }

  let duration =
      humanizeDuration(new Date(build.finishTime) - new Date(build.startTime));
  const msg = `${build.substitutions.REPO_NAME} nightly finished with status ` +
      `${build.status}, in ${duration}.`;

  await sendEmail(build, msg);
  await sendChatMsg(build, msg);
};

async function sendChatMsg(build, msg) {
  const joke = (await (await fetch('https://icanhazdadjoke.com/', {
                  headers: {'Accept': 'application/json'}
                })).json())
                   .joke;
  const jokeMsg = `Oh no! Failed builds are not fun... So here's a joke ` +
      `to brighten your day :) -- ${joke}`;
  const chatMsg = `${msg} <${build.logUrl}|See logs>. ${jokeMsg}`;
  const res = await request(process.env.HANGOUTS_URL, {
    resolveWithFullResponse: true,
    method: 'POST',
    json: true,
    body: {text: chatMsg},
  });
  console.log(`statusCode: ${res.statusCode}`);
  console.log(res.body);
}

async function sendEmail(build, msg) {
  let emailMsg = `<p>${msg}</p><p><a href="${build.logUrl}">Build logs</a></p>`;
  const email = {
    from: config.MAILGUN_FROM,
    to: config.MAILGUN_TO,
    subject: `Nightly ${build.substitutions.REPO_NAME}: ${build.status}`,
    text: emailMsg,
    html: emailMsg
  };
  await mailgun.messages().send(email);
}
