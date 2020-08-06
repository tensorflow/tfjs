/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const http = require('http');
const socketio = require('socket.io');
const fs = require('fs');
const path = require('path');

const port = process.env.PORT || 8001;
let io;
let availableBrowsers;

function setAvailableBrowsers() {
  const BrowserStack = require('browserstack');
  if (process.env.BROWSERSTACK_USERNAME == null ||
      process.env.BROWSERSTACK_ACCESS_KEY == null) {
    throw new Error(
        `Please export your BrowserStack username and access key by running` +
        `the following commands in the terminal:
          export BROWSERSTACK_USERNAME=YOUR_USERNAME
          export BROWSERSTACK_ACCESS_KEY=YOUR_ACCESS_KEY`);
  }
  const browserStackCredentials = {
    username: process.env.BROWSERSTACK_USERNAME,
    password: process.env.BROWSERSTACK_ACCESS_KEY
  };
  const automateClient =
      BrowserStack.createAutomateClient(browserStackCredentials);
  automateClient.getBrowsers((error, browsers) => {
    if (error != null) {
      console.log(error);
      throw new Error('Failed to authenticate BrowserStack.');
    } else {
      availableBrowsers = browsers;
      selectBrowsers();
      runServer();
      console.log('Successfully authenticated BrowserStack.');
    }
  });
}

function runServer() {
  const app = http.createServer((request, response) => {
    const url = request.url === '/' ? '/index.html' : request.url;
    let filePath = path.join(__dirname, url);
    if (!fs.existsSync(filePath)) {
      filePath = path.join(__dirname, '../', url);
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        response.writeHead(404);
        response.end(JSON.stringify(err));
        return;
      }
      response.writeHead(200);
      response.end(data);
    });
  });
  app.listen(port, () => {
    console.log(`  > Running socket on port: ${port}`);
  });

  io = socketio(app);
  io.on('connection', socket => {
    socket.emit('availableBrowsers', availableBrowsers);
  });
}

function selectBrowsers() {
  const selectedBrowsers = [];
  // Array `availableBrowsers` is a list of BrowserStack's supported browser
  // configurations (Automate service).
  for (const combination of availableBrowsers) {
    const androidOsVersions = ['9.0', '10.0', '8.1'];
    if (combination.os === 'android' &&
        androidOsVersions.indexOf(combination.os_version) > -1) {
      selectedBrowsers.push(combination);
    }

    const iosOsVersions = ['13', '12'];
    if (combination.os === 'ios' &&
        iosOsVersions.indexOf(combination.os_version) > -1) {
      selectedBrowsers.push(combination);
    }

    if (combination.os === 'Windows' && combination.os_version === '10') {
      if (combination.browser === 'chrome' &&
          combination.browser_version === '84.0') {
        selectedBrowsers.push(combination);
      }
      if (combination.browser === 'firefox' &&
          combination.browser_version === '79.0') {
        selectedBrowsers.push(combination);
      }
      if (combination.browser === 'edge' &&
          combination.browser_version === '84.0') {
        selectedBrowsers.push(combination);
      }
      if (combination.browser === 'ie' &&
          combination.browser_version === '11.0') {
        selectedBrowsers.push(combination);
      }
    }

    if (combination.os === 'OS X' && combination.os_version === 'Catalina') {
      if (combination.browser === 'chrome' &&
          combination.browser_version === '84.0') {
        selectedBrowsers.push(combination);
      }
      if (combination.browser === 'firefox' &&
          combination.browser_version === '79.0') {
        selectedBrowsers.push(combination);
      }
      if (combination.browser === 'safari' &&
          combination.browser_version === '13.1') {
        selectedBrowsers.push(combination);
      }
    }
  }
  console.log(`${
      selectedBrowsers
          .length} combinations are selected in ./browser_list.json`);
  fs.writeFileSync(
      'browser_list.json', JSON.stringify(selectedBrowsers, null, 2));
}

setAvailableBrowsers();
