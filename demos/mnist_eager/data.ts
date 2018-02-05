import * as dl from 'deeplearn';

const TRAIN_TEST_RATIO = 5 / 6;

const mnistConfig: dl.XhrDatasetConfig = {
  'data': [
    {
      'name': 'images',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_images.png',
      'dataType': 'png',
      'shape': [28, 28, 1]
    },
    {
      'name': 'labels',
      'path': 'https://storage.googleapis.com/learnjs-data/model-builder/' +
          'mnist_labels_uint8',
      'dataType': 'uint8',
      'shape': [10]
    }
  ],
  modelConfigs: {}
};

export class MnistData {
  private dataset: dl.XhrDataset;
  private trainingData: dl.Tensor[][];
  private testData: dl.Tensor[][];
  private trainIndices: Uint32Array;
  private testIndices: Uint32Array;

  private shuffledTrainIndex = 0;
  private shuffledTestIndex = 0;

  public nextTrainBatch(batchSize: number):
      {xs: dl.Tensor2D, labels: dl.Tensor2D} {
    return this.nextBatch(batchSize, this.trainingData, () => {
      this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
      return this.trainIndices[this.shuffledTrainIndex];
    });
  }

  public nextTestBatch(batchSize: number):
      {xs: dl.Tensor2D, labels: dl.Tensor2D} {
    return this.nextBatch(batchSize, this.testData, () => {
      this.shuffledTestIndex =
          (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  private nextBatch(
      batchSize: number, data: dl.Tensor[][],
      index: () => number): {xs: dl.Tensor2D, labels: dl.Tensor2D} {
    let xs: dl.Tensor2D = null;
    let labels: dl.Tensor2D = null;

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const x = data[0][idx].reshape([1, 784]) as dl.Tensor2D;
      xs = concatWithNulls(xs, x);

      const label = data[1][idx].reshape([1, 10]) as dl.Tensor2D;
      labels = concatWithNulls(labels, label);
    }
    return {xs, labels};
  }

  public async load() {
    this.dataset = new dl.XhrDataset(mnistConfig);
    await this.dataset.fetchData();

    this.dataset.normalizeWithinBounds(0, -1, 1);
    this.trainingData = this.getTrainingData();
    this.testData = this.getTestData();

    this.trainIndices =
        dl.util.createShuffledIndices(this.trainingData[0].length);
    this.testIndices = dl.util.createShuffledIndices(this.testData[0].length);
  }

  private getTrainingData(): dl.Tensor[][] {
    const [images, labels] =
        this.dataset.getData() as [dl.Tensor[], dl.Tensor[]];

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
  }

  private getTestData(): dl.Tensor[][] {
    const data = this.dataset.getData();
    if (data == null) {
      return null;
    }
    const [images, labels] =
        this.dataset.getData() as [dl.Tensor[], dl.Tensor[]];

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
  }
}

/**
 * TODO(nsthorat): Add math.stack, similar to np.stack, which will avoid the
 * need for us allowing concating with null values.
 */
function concatWithNulls(x1: dl.Tensor2D, x2: dl.Tensor2D): dl.Tensor2D {
  if (x1 == null && x2 == null) {
    return null;
  }
  if (x1 == null) {
    return x2;
  } else if (x2 === null) {
    return x1;
  }
  const axis = 0;
  return x1.concat(x2, axis);
}
