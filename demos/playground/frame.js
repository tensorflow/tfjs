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
'use strict';

// TODO(nsthorat): Check if this works in firefox / safari.
const htmlContainer = document.getElementById('html-container');
const htmlconsoleElement = document.getElementById('html');
const consoleElement = document.getElementById('console');
const errorConsoleElement = document.getElementById('error');

// Wait for deeplearn script to load before executing JS.
let scriptsPendingLoadCount = 1;
// JavaScript eval function if scripts are pending.
let jsEvalAwaitingScriptsFn;

const scriptLoaded = () => {
  scriptsPendingLoadCount--;
  if (scriptsPendingLoadCount === 0) {
    allScriptsLoaded();
  }
};

const deeplearnScriptElement = document.getElementById('deeplearn-script');
deeplearnScriptElement.onload = () => {
  setTimeout(scriptLoaded);
};

deeplearnScriptElement.src = 'https://unpkg.com/deeplearn';

function allScriptsLoaded() {
  if (jsEvalAwaitingScriptsFn != null) {
    jsEvalAwaitingScriptsFn();
  }
  jsEvalAwaitingScriptsFn = null;
}

window.addEventListener('message', async function (e) {
  const mainWindow = e.source;
  const result = '';

  const data = JSON.parse(e.data);
  if (data['js'] != null) {
    const executeJs = async () => {
      errorConsoleElement.innerText = '';

      window.console.clear();

      try {
        await eval(data['js']);
      } catch (e) {
        errorConsoleElement.innerText = e;
        throw e;
      }
    };
    // Don't eval javascript until all scripts from HTML are loaded.
    // This lets us load external JS modules and ensure that they're
    // ready before the javascript executes.
    if (scriptsPendingLoadCount === 0) {
      executeJs();
    } else {
      jsEvalAwaitingScriptsFn = executeJs;
    }
  } else if (data['html'] != null && data['html'] != '') {
    htmlconsoleElement.innerHTML = data['html'];

    // Find script tags, put them in the head.
    const scripts = htmlconsoleElement.getElementsByTagName('script');
    scriptsPendingLoadCount += scripts.length;

    for (let i = 0; i < scripts.length; i++) {
      const newScript = document.createElement('script');
      newScript.onload = scriptLoaded;

      newScript.src = scripts[i].src;
      document.head.appendChild(newScript);
    };

    flashOutput(htmlContainer);
  }
});

const windowLog = window.console.log;
// Override console.log to write to our console HTML element.
window.console.log = (str) => {
  consoleElement.innerHTML += str + '<br/>';
  windowLog(str);

  const consoleContainer = document.getElementById('console-container');
  flashOutput(consoleContainer);
};

window.console.clear = () => {
  consoleElement.innerText = '';
};

// Keep a map of the id to a count so only remove the class for the last ID.
const flashCounts = {};
const elements = [];
// Flash the background of the element green.
function flashOutput(elem) {
  // Find the element id in the elements array.
  let id;
  for (let i = 0; i < elements.length; i++) {
    if (elements[i] === elem) {
      id = i;
    }
  }
  if (id == null) {
    id = elements.length;
    elements.push(elem);
  }

  if (flashCounts[id] == null) {
    flashCounts[id] = 0;
  } else {
    flashCounts[id]++;
  }
  const count = flashCounts[id];

  elem.classList.add('output-container-active');
  setTimeout(() => {
    if (count === flashCounts[id]) {
      elem.classList.remove('output-container-active');
    }
  }, 200);
}
