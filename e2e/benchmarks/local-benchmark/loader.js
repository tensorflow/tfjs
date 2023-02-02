/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

// Promise is used to guarantee scripts are loaded in order.
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.onload = resolve;
    script.onerror = reject;
    script.src = url;
    if (url.startsWith('http')) {
      script.crossOrigin = 'anonymous';
    }
    document.body.append(script);
  })
}

function processUrls(urls, localBuild) {
  for (let i = 0; i < urls.length; i++) {
    let name =
        urls[i].split('/')[0].replace('tfjs-', '').replace('backend-', '');
    if (localBuild.includes(name)) {
      urls[i] = `../../../dist/bin/${urls[i]}`;
    } else {
      urls[i] =
          `https://unpkg.com/@tensorflow/${urls[i].replace('/', '@latest/')}`;
    }
  }
}

async function loadTFJS(localBuild) {
  let urls = [
    'tfjs-core/dist/tf-core.js',
    'tfjs-backend-cpu/dist/tf-backend-cpu.js',
    'tfjs-backend-webgl/dist/tf-backend-webgl.js',
    'tfjs-backend-webgpu/dist/tf-backend-webgpu.js',
    'tfjs-layers/dist/tf-layers.js',
    'tfjs-converter/dist/tf-converter.js',
    'tfjs-backend-wasm/dist/tf-backend-wasm.js',
    'tfjs-automl/dist/tf-automl.js',
  ];

  processUrls(urls, localBuild);
  urls = urls.concat([
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet@2',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/body-pix@2',
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection',
    '../model_config.js',
    '../benchmark_util.js',
    './util.js',
    './index.js',
    './dump.js',
  ]);

  for (let url of urls) {
    await loadScript(url);
  }
}
