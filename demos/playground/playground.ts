import * as dl from '../deeplearn';
import {SqueezeNet} from '../../models/squeezenet/squeezenet';

// tslint:disable-next-line:no-any
const w: any = window;

// Add all the dl exports and models to the top level window.
for (const prop in dl) {
  // tslint:disable-next-line:no-any
  w[prop] = (dl as any)[prop];
}
w['models'] = {};
w['models']['SqueezeNet'] = SqueezeNet;

const GITHUB_JS_FILENAME = 'js';
const GITHUB_HTML_FILENAME = 'html';

const saveButtonElement = document.getElementById('save');
const runButtonElement = document.getElementById('run');
const jscontentElement = document.getElementById('jscontent');
const htmlcontentElement = document.getElementById('htmlcontent');
const gistUrlElement = document.getElementById('gist-url') as HTMLInputElement;
const consoleElement = document.getElementById('console');
const htmlconsoleElement = document.getElementById('html');

saveButtonElement.addEventListener('click', async () => {
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
});

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

// Override console.log to write to our console HTML element.
window.console.log = (str: string) => {
  consoleElement.innerText += str + '\n';
};

function runHTML() {
  htmlconsoleElement.innerHTML = htmlcontentElement.innerText;
}

async function runCode() {
  runHTML();
  consoleElement.innerText = '';

  try {
    // Eval in an async() so we can directly use await.
    eval(`(async () => {
      ${jscontentElement.innerText}
    })();`);
  } catch (e) {
    const error = new Error();
    window.console.log(e.toString());
    window.console.log(error.stack);
  }
}

runButtonElement.addEventListener('click', runCode);

loadGistFromURL();
