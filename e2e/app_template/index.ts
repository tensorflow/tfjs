import * as tfconv from '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-cpu';

async function main() {
  const modelUrl = 'http://localhost:8080/model2.json';
  const model = await tfconv.loadGraphModel(modelUrl);

  model.predict(null);
}

main();
