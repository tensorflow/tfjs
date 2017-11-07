/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
const GITHUB_JS_FILENAME = 'js';
const GITHUB_HTML_FILENAME = 'html';

const saveButtonElement = document.getElementById('save');
const runButtonElement = document.getElementById('run');
const jscontentElement = document.getElementById('jscontent');
const htmlcontentElement = document.getElementById('htmlcontent');
const gistUrlElement = document.getElementById('gist-url') as HTMLInputElement;
const iframeElement = document.getElementById('sandboxed') as HTMLIFrameElement;

const saveButtonHandler = async () => {
  runCode();

  gistUrlElement.value = '...saving...';
  const jsCodeStr = jscontentElement.innerText.trim();
  const htmlCodeStr = htmlcontentElement.innerText.trim();

  // tslint:disable-next-line:no-any
  const content: any = {
    'description': 'deeplearn.js playground ' + Date.now().toString(),
    'public': true,
    'files': {}
  };

  if (jsCodeStr !== '') {
    content['files'][GITHUB_JS_FILENAME] = {'content': jsCodeStr};
  }
  if (htmlCodeStr !== '') {
    content['files'][GITHUB_HTML_FILENAME] = {'content': htmlCodeStr};
  }

  const init: RequestInit = {method: 'POST', body: JSON.stringify(content)};
  const result = await fetch('https://api.github.com/gists', init);

  const json = await result.json();

  gistUrlElement.value = json['html_url'];

  window.location.hash = `#${json['id']}`;
};

// TODO(nsthorat): bring this back once we use github logins.
// saveButtonElement.addEventListener('click', saveButtonHandler);

async function loadGistFromURL() {
  if (window.location.hash && window.location.hash !== '#') {
    gistUrlElement.value = '...loading...';

    const gistId = window.location.hash.substr(1);

    const result = await fetch('https://api.github.com/gists/' + gistId);
    const json = await result.json();
    gistUrlElement.value = json['html_url'];

    if (json['files'][GITHUB_JS_FILENAME] != null) {
      const jsFile = json['files'][GITHUB_JS_FILENAME]['raw_url'];

      const jsResult = await fetch(jsFile);
      const jsCode = await jsResult.text();

      jscontentElement.innerText = jsCode;
    }

    if (json['files'][GITHUB_HTML_FILENAME] != null) {
      const htmlFile = json['files'][GITHUB_HTML_FILENAME]['raw_url'];

      const htmlResult = await fetch(htmlFile);
      const htmlCode = await htmlResult.text();

      htmlcontentElement.innerText = htmlCode;
      runHTML();
    }

  } else {
    gistUrlElement.value = 'Unsaved';
  }
}

function runHTML() {
  iframeElement.contentWindow.postMessage(
      JSON.stringify({'html': htmlcontentElement.innerText}), '*');
}

async function runCode() {
  runHTML();

  try {
    // In an async so we can use top level await.
    const js = `(async () => {
      ${jscontentElement.innerText}
     })();`;
    iframeElement.contentWindow.postMessage(JSON.stringify({'js': js}), '*');
  } catch (e) {
    const error = new Error();
    window.console.log(e.toString());
    window.console.log(error.stack);
  }
}

runButtonElement.addEventListener('click', runCode);

loadGistFromURL();
