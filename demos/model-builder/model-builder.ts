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

import '../ndarray-image-visualizer';
import '../ndarray-logits-visualizer';
import './model-layer';
import '../demo-header';
import '../demo-footer';

// tslint:disable-next-line:max-line-length
import {AdadeltaOptimizer, AdagradOptimizer, AdamaxOptimizer, AdamOptimizer, Array1D, Array3D, DataStats, ENV, FeedEntry, Graph, GraphRunner, GraphRunnerEventObserver, InCPUMemoryShuffledInputProviderBuilder, InMemoryDataset, MetricReduction, MomentumOptimizer, NDArray, NDArrayMath, Optimizer, RMSPropOptimizer, Scalar, Session, SGDOptimizer, Tensor, util, xhr_dataset, XhrDataset, XhrDatasetConfig} from 'deeplearn';

import {NDArrayImageVisualizer} from '../ndarray-image-visualizer';
import {NDArrayLogitsVisualizer} from '../ndarray-logits-visualizer';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

import {LayerBuilder, LayerWeightsDict} from './layer_builder';
import {ModelLayer} from './model-layer';
import * as model_builder_util from './model_builder_util';

const DATASETS_CONFIG_JSON = 'model-builder-datasets-config.json';

/** How often to evaluate the model against test data. */
const EVAL_INTERVAL_MS = 1500;
/** How often to compute the cost. Downloading the cost stalls the GPU. */
const COST_INTERVAL_MS = 500;
/** How many inference examples to show when evaluating accuracy. */
const INFERENCE_EXAMPLE_COUNT = 15;
const INFERENCE_IMAGE_SIZE_PX = 100;
/**
 * How often to show inference examples. This should be less often than
 * EVAL_INTERVAL_MS as we only show inference examples during an eval.
 */
const INFERENCE_EXAMPLE_INTERVAL_MS = 3000;

// Smoothing factor for the examples/s standalone text statistic.
const EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;

const TRAIN_TEST_RATIO = 5 / 6;

const IMAGE_DATA_INDEX = 0;
const LABEL_DATA_INDEX = 1;

enum Normalization {
  NORMALIZATION_NEGATIVE_ONE_TO_ONE,
  NORMALIZATION_ZERO_TO_ONE,
  NORMALIZATION_NONE
}

// tslint:disable-next-line:variable-name
export let ModelBuilderPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'model-builder',
  properties: {
    inputShapeDisplay: String,
    isValid: Boolean,
    inferencesPerSec: Number,
    inferenceDuration: Number,
    examplesTrained: Number,
    examplesPerSec: Number,
    totalTimeSec: String,
    applicationState: Number,
    modelInitialized: Boolean,
    showTrainStats: Boolean,
    datasetDownloaded: Boolean,
    datasetNames: Array,
    selectedDatasetName: String,
    modelNames: Array,
    selectedOptimizerName: String,
    optimizerNames: Array,
    learningRate: Number,
    momentum: Number,
    needMomentum: Boolean,
    gamma: Number,
    needGamma: Boolean,
    beta1: Number,
    needBeta1: Boolean,
    beta2: Number,
    needBeta2: Boolean,
    batchSize: Number,
    selectedModelName: String,
    selectedNormalizationOption:
        {type: Number, value: Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE},
    // Stats
    showDatasetStats: Boolean,
    statsInputMin: Number,
    statsInputMax: Number,
    statsInputShapeDisplay: String,
    statsLabelShapeDisplay: String,
    statsExampleCount: Number,
  }
});

export enum ApplicationState {
  IDLE = 1,
  TRAINING = 2
}

export class ModelBuilder extends ModelBuilderPolymer {
  // Used in the html template.
  applicationState: ApplicationState;
  modelInitialized: boolean;
  showTrainStats: boolean;
  datasetDownloaded: boolean;
  modelNames: string[];
  optimizerNames: string[];
  needMomentum: boolean;
  needGamma: boolean;
  needBeta1: boolean;
  needBeta2: boolean;

  // Stats.
  showDatasetStats: boolean;
  statsInputRange: string;
  statsInputShapeDisplay: string;
  statsLabelShapeDisplay: string;
  statsExampleCount: number;
  examplesTrained: number;
  inferenceDuration: number;

  // Polymer properties.
  private isValid: boolean;
  private totalTimeSec: string;
  private selectedNormalizationOption: number;

  // Datasets and models.
  private graphRunner: GraphRunner;
  private graph: Graph;
  private session: Session;
  private optimizer: Optimizer;
  private xTensor: Tensor;
  private labelTensor: Tensor;
  private costTensor: Tensor;
  private accuracyTensor: Tensor;
  private predictionTensor: Tensor;

  private datasetNames: string[];
  private selectedDatasetName: string;
  private selectedModelName: string;
  private selectedOptimizerName: string;
  private loadedWeights: LayerWeightsDict[]|null;
  private dataSets: {[datasetName: string]: InMemoryDataset};
  private dataSet: InMemoryDataset;
  private xhrDatasetConfigs: {[datasetName: string]: XhrDatasetConfig};
  private datasetStats: DataStats[];
  private learningRate: number;
  private momentum: number;
  private gamma: number;
  private beta1: number;
  private beta2: number;
  private batchSize: number;

  // Charts.
  private costChart: Chart;
  private accuracyChart: Chart;
  private examplesPerSecChart: Chart;
  private costChartData: ChartPoint[];
  private accuracyChartData: ChartPoint[];
  private examplesPerSecChartData: ChartPoint[];

  private trainButton: HTMLButtonElement;

  // Visualizers.
  private inputNDArrayVisualizers: NDArrayImageVisualizer[];
  private outputNDArrayVisualizers: NDArrayLogitsVisualizer[];

  private inputShape: number[];
  private labelShape: number[];
  private examplesPerSec: number;
  private inferencesPerSec: number;

  private inputLayer: ModelLayer;
  private hiddenLayers: ModelLayer[];

  private layersContainer: HTMLDivElement;

  private math: NDArrayMath;

  ready() {
    this.math = ENV.math;

    this.createGraphRunner();
    this.optimizer = new MomentumOptimizer(this.learningRate, this.momentum);

    // Set up datasets.
    this.populateDatasets();

    this.querySelector('#dataset-dropdown .dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              // Update the dataset.
              const datasetName = event.detail.selected;
              this.updateSelectedDataset(datasetName);

              // TODO(nsthorat): Remember the last model used for each dataset.
              this.removeAllLayers();
            });
    this.querySelector('#model-dropdown .dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              // Update the model.
              const modelName = event.detail.selected;
              this.updateSelectedModel(modelName);
            });

    {
      const normalizationDropdown =
          this.querySelector('#normalization-dropdown .dropdown-content');
      // tslint:disable-next-line:no-any
      normalizationDropdown.addEventListener('iron-activate', (event: any) => {
        const selectedNormalizationOption = event.detail.selected;
        this.applyNormalization(selectedNormalizationOption);
        this.setupDatasetStats();
      });
    }
    this.querySelector('#optimizer-dropdown .dropdown-content')
        // tslint:disable-next-line:no-any
        .addEventListener('iron-activate', (event: any) => {
          // Activate, deactivate hyper parameter inputs.
          this.refreshHyperParamRequirements(event.detail.selected);
        });
    this.learningRate = 0.1;
    this.momentum = 0.1;
    this.needMomentum = true;
    this.gamma = 0.1;
    this.needGamma = false;
    this.beta1 = 0.9;
    this.needBeta1 = false;
    this.beta2 = 0.999;
    this.needBeta2 = false;
    this.batchSize = 64;
    // Default optimizer is momentum
    this.selectedOptimizerName = 'momentum';
    this.optimizerNames =
        ['sgd', 'momentum', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax'];

    this.applicationState = ApplicationState.IDLE;
    this.loadedWeights = null;
    this.modelInitialized = false;
    this.showTrainStats = false;
    this.showDatasetStats = false;

    const addButton = this.querySelector('#add-layer');
    addButton.addEventListener('click', () => this.addLayer());

    const downloadModelButton = this.querySelector('#download-model');
    downloadModelButton.addEventListener('click', () => this.downloadModel());
    const uploadModelButton = this.querySelector('#upload-model');
    uploadModelButton.addEventListener('click', () => this.uploadModel());
    this.setupUploadModelButton();

    const uploadWeightsButton = this.querySelector('#upload-weights');
    uploadWeightsButton.addEventListener('click', () => this.uploadWeights());
    this.setupUploadWeightsButton();

    const stopButton = this.querySelector('#stop');
    stopButton.addEventListener('click', () => {
      this.applicationState = ApplicationState.IDLE;
      this.graphRunner.stopTraining();
    });

    this.trainButton = this.querySelector('#train') as HTMLButtonElement;
    this.trainButton.addEventListener('click', () => {
      this.createModel();
      this.startTraining();
    });

    this.querySelector('#environment-toggle')
        .addEventListener('change', (event) => {
          this.math =
              // tslint:disable-next-line:no-any
              (event.target as any).active ? this.mathGPU : this.mathCPU;
          this.graphRunner.setMath(this.math);
        });

    this.hiddenLayers = [];
    this.examplesPerSec = 0;
    this.inferencesPerSec = 0;
  }

  createGraphRunner() {
    const eventObserver: GraphRunnerEventObserver = {
      batchesTrainedCallback: (batchesTrained: number) =>
          this.displayBatchesTrained(batchesTrained),
      avgCostCallback: (avgCost: Scalar) => this.displayCost(avgCost),
      metricCallback: (metric: Scalar) => this.displayAccuracy(metric),
      inferenceExamplesCallback:
          (inputFeeds: FeedEntry[][], inferenceOutputs: NDArray[]) =>
              this.displayInferenceExamplesOutput(inputFeeds, inferenceOutputs),
      inferenceExamplesPerSecCallback: (examplesPerSec: number) =>
          this.displayInferenceExamplesPerSec(examplesPerSec),
      trainExamplesPerSecCallback: (examplesPerSec: number) =>
          this.displayExamplesPerSec(examplesPerSec),
      totalTimeCallback: (totalTimeSec: number) => this.totalTimeSec =
          totalTimeSec.toFixed(1),
    };
    this.graphRunner = new GraphRunner(this.math, this.session, eventObserver);
  }

  isTraining(applicationState: ApplicationState): boolean {
    return applicationState === ApplicationState.TRAINING;
  }

  isIdle(applicationState: ApplicationState): boolean {
    return applicationState === ApplicationState.IDLE;
  }

  private getTestData(): NDArray[][] {
    const data = this.dataSet.getData();
    if (data == null) {
      return null;
    }
    const [images, labels] = this.dataSet.getData() as [NDArray[], NDArray[]];

    const start = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(start), labels.slice(start)];
  }

  private getTrainingData(): NDArray[][] {
    const [images, labels] = this.dataSet.getData() as [NDArray[], NDArray[]];

    const end = Math.floor(TRAIN_TEST_RATIO * images.length);

    return [images.slice(0, end), labels.slice(0, end)];
  }

  private startInference() {
    const testData = this.getTestData();
    if (testData == null) {
      // Dataset not ready yet.
      return;
    }
    if (this.isValid && (testData != null)) {
      const inferenceShuffledInputProviderGenerator =
          new InCPUMemoryShuffledInputProviderBuilder(testData);
      const [inferenceInputProvider, inferenceLabelProvider] =
          inferenceShuffledInputProviderGenerator.getInputProviders();

      const inferenceFeeds = [
        {tensor: this.xTensor, data: inferenceInputProvider},
        {tensor: this.labelTensor, data: inferenceLabelProvider}
      ];

      this.graphRunner.infer(
          this.predictionTensor, inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS,
          INFERENCE_EXAMPLE_COUNT);
    }
  }

  private resetHyperParamRequirements() {
    this.needMomentum = false;
    this.needGamma = false;
    this.needBeta1 = false;
    this.needBeta2 = false;
  }

  /**
   * Set flag to disable input by optimizer selection.
   */
  private refreshHyperParamRequirements(optimizerName: string) {
    this.resetHyperParamRequirements();
    switch (optimizerName) {
      case 'sgd': {
        // No additional hyper parameters
        break;
      }
      case 'momentum': {
        this.needMomentum = true;
        break;
      }
      case 'rmsprop': {
        this.needMomentum = true;
        this.needGamma = true;
        break;
      }
      case 'adagrad': {
        break;
      }
      case 'adadelta': {
        this.needGamma = true;
        break;
      }
      case 'adam': {
        this.needBeta1 = true;
        this.needBeta2 = true;
        break;
      }
      case 'adamax': {
        this.needBeta1 = true;
        this.needBeta2 = true;
        break;
      }
      default: {
        throw new Error(`Unknown optimizer "${this.selectedOptimizerName}"`);
      }
    }
  }

  private createOptimizer() {
    switch (this.selectedOptimizerName) {
      case 'sgd': {
        return new SGDOptimizer(+this.learningRate);
      }
      case 'momentum': {
        return new MomentumOptimizer(+this.learningRate, +this.momentum);
      }
      case 'rmsprop': {
        return new RMSPropOptimizer(+this.learningRate, +this.gamma);
      }
      case 'adagrad': {
        return new AdagradOptimizer(+this.learningRate);
      }
      case 'adadelta': {
        return new AdadeltaOptimizer(+this.learningRate, +this.gamma);
      }
      case 'adam': {
        return new AdamOptimizer(+this.learningRate, +this.beta1, +this.beta2);
      }
      case 'adamax': {
        return new AdamaxOptimizer(
            +this.learningRate, +this.beta1, +this.beta2);
      }
      default: {
        throw new Error(`Unknown optimizer "${this.selectedOptimizerName}"`);
      }
    }
  }

  private startTraining() {
    const trainingData = this.getTrainingData();
    const testData = this.getTestData();

    // Recreate optimizer with the selected optimizer and hyperparameters.
    this.optimizer = this.createOptimizer();

    if (this.isValid && (trainingData != null) && (testData != null)) {
      this.recreateCharts();
      this.graphRunner.resetStatistics();

      const trainingShuffledInputProviderGenerator =
          new InCPUMemoryShuffledInputProviderBuilder(trainingData);
      const [trainInputProvider, trainLabelProvider] =
          trainingShuffledInputProviderGenerator.getInputProviders();

      const trainFeeds = [
        {tensor: this.xTensor, data: trainInputProvider},
        {tensor: this.labelTensor, data: trainLabelProvider}
      ];

      const accuracyShuffledInputProviderGenerator =
          new InCPUMemoryShuffledInputProviderBuilder(testData);
      const [accuracyInputProvider, accuracyLabelProvider] =
          accuracyShuffledInputProviderGenerator.getInputProviders();

      const accuracyFeeds = [
        {tensor: this.xTensor, data: accuracyInputProvider},
        {tensor: this.labelTensor, data: accuracyLabelProvider}
      ];

      this.graphRunner.train(
          this.costTensor, trainFeeds, this.batchSize, this.optimizer,
          undefined /** numBatches */, this.accuracyTensor, accuracyFeeds,
          this.batchSize, MetricReduction.MEAN, EVAL_INTERVAL_MS,
          COST_INTERVAL_MS);

      this.showTrainStats = true;
      this.applicationState = ApplicationState.TRAINING;
    }
  }

  private createModel() {
    if (this.session != null) {
      this.session.dispose();
    }

    this.modelInitialized = false;
    if (this.isValid === false) {
      return;
    }

    this.graph = new Graph();
    const g = this.graph;
    this.xTensor = g.placeholder('input', this.inputShape);
    this.labelTensor = g.placeholder('label', this.labelShape);

    let network = this.xTensor;

    for (let i = 0; i < this.hiddenLayers.length; i++) {
      let weights: LayerWeightsDict|null = null;
      if (this.loadedWeights != null) {
        weights = this.loadedWeights[i];
      }
      network = this.hiddenLayers[i].addLayer(g, network, i, weights);
    }
    this.predictionTensor = network;
    this.costTensor =
        g.softmaxCrossEntropyCost(this.predictionTensor, this.labelTensor);
    this.accuracyTensor =
        g.argmaxEquals(this.predictionTensor, this.labelTensor);

    this.loadedWeights = null;

    this.session = new Session(g, this.math);
    this.graphRunner.setSession(this.session);

    this.startInference();

    this.modelInitialized = true;
  }

  private populateDatasets() {
    this.dataSets = {};
    xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
        .then(
            xhrDatasetConfigs => {
              for (const datasetName in xhrDatasetConfigs) {
                if (xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                  this.dataSets[datasetName] =
                      new XhrDataset(xhrDatasetConfigs[datasetName]);
                }
              }
              this.datasetNames = Object.keys(this.dataSets);
              this.selectedDatasetName = this.datasetNames[0];
              this.xhrDatasetConfigs = xhrDatasetConfigs;
              this.updateSelectedDataset(this.datasetNames[0]);
            },
            error => {
              throw new Error(`Dataset config could not be loaded: ${error}`);
            });
  }

  private updateSelectedDataset(datasetName: string) {
    if (this.dataSet != null) {
      this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
    }

    this.graphRunner.stopTraining();
    this.graphRunner.stopInferring();

    if (this.dataSet != null) {
      this.dataSet.dispose();
    }

    this.selectedDatasetName = datasetName;
    this.selectedModelName = '';
    this.dataSet = this.dataSets[datasetName];
    this.datasetDownloaded = false;
    this.showDatasetStats = false;

    this.dataSet.fetchData().then(() => {
      this.datasetDownloaded = true;
      this.applyNormalization(this.selectedNormalizationOption);
      this.setupDatasetStats();
      if (this.isValid) {
        this.createModel();
      }
      // Get prebuilt models.
      this.populateModelDropdown();
    });

    this.inputShape = this.dataSet.getDataShape(IMAGE_DATA_INDEX);
    this.labelShape = this.dataSet.getDataShape(LABEL_DATA_INDEX);

    this.layersContainer =
        this.querySelector('#hidden-layers') as HTMLDivElement;

    this.inputLayer = this.querySelector('#input-layer') as ModelLayer;
    this.inputLayer.outputShapeDisplay =
        model_builder_util.getDisplayShape(this.inputShape);

    const labelShapeDisplay =
        model_builder_util.getDisplayShape(this.labelShape);
    const costLayer = this.querySelector('#cost-layer') as ModelLayer;
    costLayer.inputShapeDisplay = labelShapeDisplay;
    costLayer.outputShapeDisplay = labelShapeDisplay;

    const outputLayer = this.querySelector('#output-layer') as ModelLayer;
    outputLayer.inputShapeDisplay = labelShapeDisplay;

    // Setup the inference example container.
    // TODO(nsthorat): Generalize this.
    const inferenceContainer =
        this.querySelector('#inference-container') as HTMLElement;
    inferenceContainer.innerHTML = '';
    this.inputNDArrayVisualizers = [];
    this.outputNDArrayVisualizers = [];
    for (let i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
      const inferenceExampleElement = document.createElement('div');
      inferenceExampleElement.className = 'inference-example';

      // Set up the input visualizer.
      const ndarrayImageVisualizer =
          document.createElement('ndarray-image-visualizer') as
          NDArrayImageVisualizer;
      ndarrayImageVisualizer.setShape(this.inputShape);
      ndarrayImageVisualizer.setSize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.inputNDArrayVisualizers.push(ndarrayImageVisualizer);
      inferenceExampleElement.appendChild(ndarrayImageVisualizer);

      // Set up the output ndarray visualizer.
      const ndarrayLogitsVisualizer =
          document.createElement('ndarray-logits-visualizer') as
          NDArrayLogitsVisualizer;
      ndarrayLogitsVisualizer.initialize(
          INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
      this.outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
      inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);

      inferenceContainer.appendChild(inferenceExampleElement);
    }
  }

  private populateModelDropdown() {
    const modelNames = ['Custom'];

    const modelConfigs =
        this.xhrDatasetConfigs[this.selectedDatasetName].modelConfigs;
    for (const modelName in modelConfigs) {
      if (modelConfigs.hasOwnProperty(modelName)) {
        modelNames.push(modelName);
      }
    }
    this.modelNames = modelNames;
    this.selectedModelName = modelNames[modelNames.length - 1];
    this.updateSelectedModel(this.selectedModelName);
  }

  private updateSelectedModel(modelName: string) {
    this.removeAllLayers();
    if (modelName === 'Custom') {
      // TODO(nsthorat): Remember the custom layers.
      return;
    }

    this.loadModelFromPath(this.xhrDatasetConfigs[this.selectedDatasetName]
                               .modelConfigs[modelName]
                               .path);
  }

  private loadModelFromPath(modelPath: string) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', modelPath);

    xhr.onload = () => {
      this.loadModelFromJson(xhr.responseText);
    };
    xhr.onerror = (error) => {
      throw new Error(`Model could not be fetched from ${modelPath}: ${error}`);
    };
    xhr.send();
  }

  private setupDatasetStats() {
    this.datasetStats = this.dataSet.getStats();
    this.statsExampleCount = this.datasetStats[IMAGE_DATA_INDEX].exampleCount;
    this.statsInputRange =
        `[${this.datasetStats[IMAGE_DATA_INDEX].inputMin}, ` +
        `${this.datasetStats[IMAGE_DATA_INDEX].inputMax}]`;
    this.statsInputShapeDisplay = model_builder_util.getDisplayShape(
        this.datasetStats[IMAGE_DATA_INDEX].shape);
    this.statsLabelShapeDisplay = model_builder_util.getDisplayShape(
        this.datasetStats[LABEL_DATA_INDEX].shape);
    this.showDatasetStats = true;
  }

  private applyNormalization(selectedNormalizationOption: number) {
    switch (selectedNormalizationOption) {
      case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE: {
        this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
        break;
      }
      case Normalization.NORMALIZATION_ZERO_TO_ONE: {
        this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
        break;
      }
      case Normalization.NORMALIZATION_NONE: {
        this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
        break;
      }
      default: { throw new Error('Normalization option must be 0, 1, or 2'); }
    }
    this.setupDatasetStats();
  }

  private recreateCharts() {
    this.costChartData = [];
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    this.costChart =
        this.createChart('cost-chart', 'Cost', this.costChartData, 0);

    if (this.accuracyChart != null) {
      this.accuracyChart.destroy();
    }
    this.accuracyChartData = [];
    this.accuracyChart = this.createChart(
        'accuracy-chart', 'Accuracy', this.accuracyChartData, 0, 100);

    if (this.examplesPerSecChart != null) {
      this.examplesPerSecChart.destroy();
    }
    this.examplesPerSecChartData = [];
    this.examplesPerSecChart = this.createChart(
        'examplespersec-chart', 'Examples/sec', this.examplesPerSecChartData,
        0);
  }

  private createChart(
      canvasId: string, label: string, data: ChartData[], min?: number,
      max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
                        .getContext('2d') as CanvasRenderingContext2D;
    return new Chart(context, {
      type: 'line',
      data: {
        datasets: [{
          data,
          fill: false,
          label,
          pointRadius: 0,
          borderColor: 'rgba(75,192,192,1)',
          borderWidth: 1,
          lineTension: 0,
          pointHitRadius: 8
        }]
      },
      options: {
        animation: {duration: 0},
        responsive: false,
        scales: {
          xAxes: [{type: 'linear', position: 'bottom'}],
          yAxes: [{
            ticks: {
              max,
              min,
            }
          }]
        }
      }
    });
  }

  displayBatchesTrained(totalBatchesTrained: number) {
    this.examplesTrained = this.batchSize * totalBatchesTrained;
  }

  displayCost(avgCost: Scalar) {
    this.costChartData.push(
        {x: this.graphRunner.getTotalBatchesTrained(), y: avgCost.get()});
    this.costChart.update();
  }

  displayAccuracy(accuracy: Scalar) {
    this.accuracyChartData.push({
      x: this.graphRunner.getTotalBatchesTrained(),
      y: accuracy.get() * 100
    });
    this.accuracyChart.update();
  }

  displayInferenceExamplesPerSec(examplesPerSec: number) {
    this.inferencesPerSec =
        this.smoothExamplesPerSec(this.inferencesPerSec, examplesPerSec);
    this.inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));
  }

  displayExamplesPerSec(examplesPerSec: number) {
    this.examplesPerSecChartData.push(
        {x: this.graphRunner.getTotalBatchesTrained(), y: examplesPerSec});
    this.examplesPerSecChart.update();
    this.examplesPerSec =
        this.smoothExamplesPerSec(this.examplesPerSec, examplesPerSec);
  }

  private smoothExamplesPerSec(
      lastExamplesPerSec: number, nextExamplesPerSec: number): number {
    return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
                   (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
                      .toPrecision(3));
  }

  displayInferenceExamplesOutput(
      inputFeeds: FeedEntry[][], inferenceOutputs: NDArray[]) {
    let images: Array3D[] = [];
    const logits: Array1D[] = [];
    const labels: Array1D[] = [];
    for (let i = 0; i < inputFeeds.length; i++) {
      images.push(inputFeeds[i][IMAGE_DATA_INDEX].data as Array3D);
      labels.push(inputFeeds[i][LABEL_DATA_INDEX].data as Array1D);
      logits.push(inferenceOutputs[i] as Array1D);
    }

    images =
        this.dataSet.unnormalizeExamples(images, IMAGE_DATA_INDEX) as Array3D[];

    // Draw the images.
    for (let i = 0; i < inputFeeds.length; i++) {
      this.inputNDArrayVisualizers[i].saveImageDataFromNDArray(images[i]);
    }

    // Draw the logits.
    for (let i = 0; i < inputFeeds.length; i++) {
      const softmaxLogits = this.math.softmax(logits[i]).asType('float32');

      this.outputNDArrayVisualizers[i].drawLogits(
          softmaxLogits, labels[i],
          this.xhrDatasetConfigs[this.selectedDatasetName].labelClassNames);
      this.inputNDArrayVisualizers[i].draw();

      softmaxLogits.dispose();
    }
  }

  addLayer(): ModelLayer {
    const modelLayer = document.createElement('model-layer') as ModelLayer;
    modelLayer.className = 'layer';
    this.layersContainer.appendChild(modelLayer);

    const lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
    const lastOutputShape = lastHiddenLayer != null ?
        lastHiddenLayer.getOutputShape() :
        this.inputShape;
    this.hiddenLayers.push(modelLayer);
    modelLayer.initialize(this, lastOutputShape);
    return modelLayer;
  }

  removeLayer(modelLayer: ModelLayer) {
    this.layersContainer.removeChild(modelLayer);
    this.hiddenLayers.splice(this.hiddenLayers.indexOf(modelLayer), 1);
    this.layerParamChanged();
  }

  private removeAllLayers() {
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      this.layersContainer.removeChild(this.hiddenLayers[i]);
    }
    this.hiddenLayers = [];
    this.layerParamChanged();
  }

  private validateModel() {
    let valid = true;
    for (let i = 0; i < this.hiddenLayers.length; ++i) {
      valid = valid && this.hiddenLayers[i].isValid();
    }
    if (this.hiddenLayers.length > 0) {
      const lastLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
      valid = valid &&
          util.arraysEqual(this.labelShape, lastLayer.getOutputShape());
    }
    this.isValid = valid && (this.hiddenLayers.length > 0);
  }

  layerParamChanged() {
    // Go through each of the model layers and propagate shapes.
    let lastOutputShape = this.inputShape;
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
    }
    this.validateModel();

    if (this.isValid) {
      this.createModel();
    }
  }

  private downloadModel() {
    const modelJson = this.getModelAsJson();
    const blob = new Blob([modelJson], {type: 'text/json'});
    const textFile = window.URL.createObjectURL(blob);

    // Force a download.
    const a = document.createElement('a');
    document.body.appendChild(a);
    a.style.display = 'none';
    a.href = textFile;
    // tslint:disable-next-line:no-any
    (a as any).download = this.selectedDatasetName + '_model';
    a.click();

    document.body.removeChild(a);
    window.URL.revokeObjectURL(textFile);
  }

  private uploadModel() {
    (this.querySelector('#model-file') as HTMLInputElement).click();
  }

  private setupUploadModelButton() {
    // Show and setup the load view button.
    const fileInput = this.querySelector('#model-file') as HTMLInputElement;
    fileInput.addEventListener('change', event => {
      const file = fileInput.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      fileInput.value = '';
      const fileReader = new FileReader();
      fileReader.onload = (evt) => {
        this.removeAllLayers();
        const modelJson: string = fileReader.result;
        this.loadModelFromJson(modelJson);
      };
      fileReader.readAsText(file);
    });
  }

  private getModelAsJson(): string {
    const layerBuilders: LayerBuilder[] = [];
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      layerBuilders.push(this.hiddenLayers[i].layerBuilder);
    }
    return JSON.stringify(layerBuilders);
  }

  private loadModelFromJson(modelJson: string) {
    let lastOutputShape = this.inputShape;

    const layerBuilders = JSON.parse(modelJson) as LayerBuilder[];
    for (let i = 0; i < layerBuilders.length; i++) {
      const modelLayer = this.addLayer();
      modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
      lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
    }
    this.validateModel();
  }

  private uploadWeights() {
    (this.querySelector('#weights-file') as HTMLInputElement).click();
  }

  private setupUploadWeightsButton() {
    // Show and setup the load view button.
    const fileInput = this.querySelector('#weights-file') as HTMLInputElement;
    fileInput.addEventListener('change', event => {
      const file = fileInput.files[0];
      // Clear out the value of the file chooser. This ensures that if the user
      // selects the same file, we'll re-read it.
      fileInput.value = '';
      const fileReader = new FileReader();
      fileReader.onload = (evt) => {
        const weightsJson: string = fileReader.result;
        this.loadWeightsFromJson(weightsJson);
        this.createModel();
      };
      fileReader.readAsText(file);
    });
  }

  private loadWeightsFromJson(weightsJson: string) {
    this.loadedWeights = JSON.parse(weightsJson) as LayerWeightsDict[];
  }
}

document.registerElement(ModelBuilder.prototype.is, ModelBuilder);
