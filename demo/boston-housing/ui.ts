/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import renderChart from 'vega-embed';
import {VisualizationSpec} from 'vega-embed';

import {linearRegressionModel, multiLayerPerceptronRegressionModel, run} from './index';

const statusElement = document.getElementById('status') as HTMLTextAreaElement;
export const updateStatus = (message: string) => {
  statusElement.value = message;
};

const baselineStatusElement =
    document.getElementById('baselineStatus') as HTMLTextAreaElement;
export const updateBaselineStatus = (message: string) => {
  baselineStatusElement.value = message;
};

export const setup = async () => {
  const trainSimpleLinearRegression = document.getElementById('simple-mlr');
  const trainNeuralNetworkLinearRegression = document.getElementById('nn-mlr');

  trainSimpleLinearRegression.addEventListener('click', async (e) => {
    const model = linearRegressionModel();
    losses = [{}];
    await run(model);
  }, false);

  trainNeuralNetworkLinearRegression.addEventListener('click', async (e) => {
    const model = multiLayerPerceptronRegressionModel();
    losses = [{}];
    await run(model);
  }, false);
};

let losses = [{}];
export const plotData =
    async (epoch: number, trainLoss: number, valLoss: number) => {
  losses.push({'epoch': epoch, 'loss': trainLoss, 'split': 'Train Loss'});
  losses.push({'epoch': epoch, 'loss': valLoss, 'split': 'Validation Loss'});

  const spec = {
    '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
    'width': 250,
    'height': 250,
    'data': {'values': losses},
    'mark': 'line',
    'encoding': {
      'x': {'field': 'epoch', 'type': 'quantitative'},
      'y': {'field': 'loss', 'type': 'quantitative'},
      'color': {'field': 'split', 'type': 'nominal'}
    }
  } as VisualizationSpec;

  return renderChart('#plot', spec, {actions: false});
};
