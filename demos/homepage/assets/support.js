/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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
function isWebGLEnabled() {
  var canvas = document.createElement('canvas');

  var attributes = {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    depth: false,
    stencil: false,
    failIfMajorPerformanceCaveat: true
  };
  return null != (canvas.getContext('webgl', attributes) ||
                  canvas.getContext('experimental-webgl', attributes));
}

function buildAndShowDialog(title, content) {
  var dialogContainer = document.createElement('div');
  dialogContainer.innerHTML = `
    <dialog id="dialog" class="mdl-dialog">
      <h4 class="mdl-dialog__title">${title}</h4>
      <div class="mdl-dialog__content">
        <p>${content}</p>
      </div>
    </dialog>
  `;
  document.body.appendChild(dialogContainer);
  var dialog = document.getElementById('dialog');
  dialog.style.width = '430px';
  dialogPolyfill.registerDialog(dialog);
  dialog.showModal();
}

function inializePolymerPage() {
  document.addEventListener('WebComponentsReady', function(event) {
    if (!isWebGLEnabled()) {
      const title = `Check if hardware acceleration is enabled.`;
      const content = `
        Looks like your device is supported but settings aren't in place.
        Please check if <b>WebGL</b> is enabled for your browser.

        See: <a href='https://superuser.com/a/836833' target='_blank'>How can I enable WebGL in my browser?</a>
      `;
      buildAndShowDialog(title, content);
    } else {
      var bundleScript = document.createElement('script');
      bundleScript.src = 'bundle.js';
      document.head.appendChild(bundleScript);
    }
  });
}
inializePolymerPage();
