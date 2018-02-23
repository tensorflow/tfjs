import * as dl from 'deeplearn';
import {MnistData} from './data';

// Hyperparameters.
const LEARNING_RATE = .1;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = dl.train.sgd(LEARNING_RATE);

// Variables that we want to optimize
const conv1OutputDepth = 8;
const conv1Weights = dl.variable(
    dl.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1) as dl.Tensor4D);

const conv2InputDepth = conv1OutputDepth;
const conv2OutputDepth = 16;
const conv2Weights = dl.variable(
    dl.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1) as
    dl.Tensor4D);

const fullyConnectedWeights = dl.variable(
    dl.randomNormal(
        [7 * 7 * conv2OutputDepth, LABELS_SIZE], 0,
        1 / Math.sqrt(7 * 7 * conv2OutputDepth)) as dl.Tensor2D);
const fullyConnectedBias = dl.variable(dl.zeros([LABELS_SIZE]) as dl.Tensor1D);

// Loss function
function loss(labels: dl.Tensor2D, ys: dl.Tensor2D) {
  return dl.losses.softmaxCrossEntropy(labels, ys).mean() as dl.Scalar;
}

// Our actual model
function model(inputXs: dl.Tensor2D): dl.Tensor2D {
  const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1);

  const strides = 2;
  const pad = 0;

  // Conv 1
  const layer1 = dl.tidy(() => {
    return xs.conv2d(conv1Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // Conv 2
  const layer2 = dl.tidy(() => {
    return layer1.conv2d(conv2Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // Final layer
  return layer2.as2D(-1, fullyConnectedWeights.shape[0])
      .matMul(fullyConnectedWeights)
      .add(fullyConnectedBias);
}

// Train the model.
export async function train(data: MnistData, log: (message: string) => void) {
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    const cost = optimizer.minimize(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      return loss(batch.labels, model(batch.xs));
    }, returnCost);

    log(`loss[${i}]: ${cost.dataSync()}`);

    await dl.nextFrame();
  }
}

// Predict the digit number from a batch of input images.
export function predict(x: dl.Tensor2D): number[] {
  const pred = dl.tidy(() => {
    const axis = 1;
    return model(x).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y: dl.Tensor2D): number[] {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
