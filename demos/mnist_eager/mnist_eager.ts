import {MnistData} from './data';
import * as model from './model';
import * as ui from './ui';

let data: MnistData;
async function load() {
  data = new MnistData();
  await data.load();
}

async function train() {
  ui.isTraining();
  await model.train(data, ui.trainingLog);
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const predictions = model.predict(batch.xs);
  const labels = model.classesFromLabel(batch.labels);

  ui.showTestResults(batch, predictions, labels);
}

async function mnist() {
  await load();
  await train();
  test();
}
mnist();
