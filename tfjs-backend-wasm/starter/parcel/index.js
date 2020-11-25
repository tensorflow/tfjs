import '@tensorflow/tfjs-backend-webgl';

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

const IMAGE_SIZE = 224;
const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

let global_model = null;
async function run() {
  tf.ENV.set('WEBGL_PACK', false);
  await tf.setBackend('webgl');
  const modelUrl = 'http://localhost:8080/tmp_web_model/model.json';
  global_model = await tfconv.loadGraphModel(modelUrl);
}
run();

async function predict(imgElement) {
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const result = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return global_model.predict(batched);
  });

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log(totalTime1);
  console.log(totalTime2);

  await showResults(imgElement, result);
}

async function showResults(imgElement, result) {
  const canvas = document.getElementById('result');
  const [batch, height, width, channel] = result.shape;
  const resultReshaped = result.reshape([height, width, channel]);
  const offset = tf.scalar(127.5);
  const denormalized = resultReshaped.mul(offset).add(offset).cast('int32');
  await tf.browser.toPixels(denormalized, canvas);
}
