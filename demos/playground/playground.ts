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
import '../demo-header';
import '../demo-footer';

const GITHUB_JS_FILENAME = 'js';
const GITHUB_HTML_FILENAME = 'html';

const saveButtonElement = document.getElementById('save');
const runButtonElement = document.getElementById('run');
const gistUrlElement = document.getElementById('gist-url') as HTMLInputElement;
const iframeElement = document.getElementById('sandboxed') as HTMLIFrameElement;

// tslint:disable-next-line:no-any
const w = window as any;
// tslint:disable-next-line:no-any
let jsEditor: any;
// tslint:disable-next-line:no-any
let htmlEditor: any;

// tslint:disable-next-line:no-any
const setupCommonEditorSettings = (editor: any) => {
  editor.setTheme('ace/theme/tomorrow');
  editor.setOptions({maxLines: Infinity});
  editor.getSession().setTabSize(2);
  editor.getSession().setUseWorker(false);
};

const loadPage = () => {
  jsEditor = w.ace.edit('jscontent');
  jsEditor.getSession().setMode('ace/mode/javascript');

  htmlEditor = w.ace.edit('htmlcontent');
  htmlEditor.getSession().setMode('ace/mode/html');

  setupCommonEditorSettings(jsEditor);
  setupCommonEditorSettings(htmlEditor);
};

const saveButtonHandler = async () => {
  runCode();

  gistUrlElement.value = '...saving...';
  const jsCodeStr = jsEditor.getValue();
  const htmlCodeStr = htmlEditor.getValue();

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
if (saveButtonElement != null) {
  saveButtonElement.addEventListener(
      'click', saveButtonHandler == null ? saveButtonHandler : () => {});
}

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

      jsEditor.setValue(jsCode, -1);
    }

    if (json['files'][GITHUB_HTML_FILENAME] != null) {
      const htmlFile = json['files'][GITHUB_HTML_FILENAME]['raw_url'];

      const htmlResult = await fetch(htmlFile);
      const htmlCode = await htmlResult.text();

      htmlEditor.setValue(htmlCode, -1);
    }

    if (w.iframeLoaded === true) {
      runCode();
    } else {
      iframeElement.addEventListener('load', () => {
        runCode();
      });
    }
  } else {
    gistUrlElement.value = 'Unsaved';
  }
}

function runHTML() {
  iframeElement.contentWindow.postMessage(
      JSON.stringify({'html': htmlEditor.getValue()}), '*');
}

async function runCode() {
  runHTML();

  try {
    // In an async so we can use top level await.
    const js = `(async () => {
      ${jsEditor.getValue()}
     })();`;
    iframeElement.contentWindow.postMessage(JSON.stringify({'js': js}), '*');
  } catch (e) {
    const error = new Error();
    window.console.log(e.toString());
    window.console.log(error.stack);
  }
}

loadPage();
loadGistFromURL();

runButtonElement.addEventListener('click', runCode);
