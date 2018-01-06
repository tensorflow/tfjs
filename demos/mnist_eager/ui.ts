import * as dl from 'deeplearn';

const math = dl.ENV.math;

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

export function isTraining() {
  statusElement.innerText = 'Training...';
}
export function trainingLog(message: string) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(
    batch: {xs: dl.Array2D<'float32'>, labels: dl.Array2D<'float32'>},
    predictions: number[], labels: number[]) {
  statusElement.innerText = 'Testing...';

  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = math.slice2D(batch.xs, [i, 0], [1, batch.xs.shape[1]]) as
        dl.Array2D<'float32'>;

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;
    if (correct) {
      totalCorrect++;
    }

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${prediction}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }

  const accuracy = 100 * totalCorrect / testExamples;
  const displayStr =
      `accuracy: ${accuracy.toFixed(2)}% (${totalCorrect} / ${testExamples})`;
  messageElement.innerText = `${displayStr}\n`;
  console.log(displayStr);
}

export function draw(image: dl.Array1D, canvas: HTMLCanvasElement) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
