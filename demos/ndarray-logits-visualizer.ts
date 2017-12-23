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

import {Array1D, ENV} from 'deeplearn';

import {PolymerElement, PolymerHTMLElement} from './polymer-spec';

const TOP_K = 3;

// tslint:disable-next-line
export let NDArrayLogitsVisualizerPolymer: new () => PolymerHTMLElement =
    PolymerElement({is: 'ndarray-logits-visualizer', properties: {}});

export class NDArrayLogitsVisualizer extends NDArrayLogitsVisualizerPolymer {
  // Polymer properties.
  logitLabelElements: HTMLElement[];
  logitVizElements: HTMLElement[];
  width: number;

  initialize(width: number, height: number) {
    this.width = width;
    this.logitLabelElements = [];
    this.logitVizElements = [];
    const container = this.querySelector('.logits-container') as HTMLElement;
    container.style.height = `${height}px`;

    for (let i = 0; i < TOP_K; i++) {
      const logitContainer = document.createElement('div');
      logitContainer.style.height = `${height / (TOP_K + 1)}px`;
      logitContainer.style.margin =
          `${height / ((2 * TOP_K) * (TOP_K + 1))}px 0`;
      logitContainer.className =
          'single-logit-container ndarray-logits-visualizer';

      const logitLabelElement = document.createElement('div');
      logitLabelElement.className = 'logit-label ndarray-logits-visualizer';
      this.logitLabelElements.push(logitLabelElement);

      const logitVizOuterElement = document.createElement('div');
      logitVizOuterElement.className =
          'logit-viz-outer ndarray-logits-visualizer';

      const logitVisInnerElement = document.createElement('div');
      logitVisInnerElement.className =
          'logit-viz-inner ndarray-logits-visualizer';
      logitVisInnerElement.innerHTML = '&nbsp;';
      logitVizOuterElement.appendChild(logitVisInnerElement);

      this.logitVizElements.push(logitVisInnerElement);

      logitContainer.appendChild(logitLabelElement);
      logitContainer.appendChild(logitVizOuterElement);
      container.appendChild(logitContainer);
    }
  }

  drawLogits(
      predictedLogits: Array1D<'float32'>, labelLogits: Array1D,
      labelClassNames?: string[]) {
    const labelClass = ENV.math.argMax(labelLogits).get();

    const topk = this.topK(predictedLogits.dataSync(), TOP_K);
    const topkIndices = topk.indices;
    const topkValues = topk.values;

    for (let i = 0; i < topkIndices.length; i++) {
      const index = topkIndices[i];
      this.logitLabelElements[i].innerText =
          labelClassNames ? labelClassNames[index] : index.toString();
      this.logitLabelElements[i].style.width =
          labelClassNames != null ? '100px' : '20px';
      this.logitVizElements[i].style.backgroundColor = index === labelClass ?
          'rgba(120, 185, 50, .84)' :
          'rgba(220, 10, 10, 0.84)';
      this.logitVizElements[i].style.width =
          `${Math.floor(100 * topkValues[i])}%`;
      this.logitVizElements[i].innerText =
          `${(100 * topkValues[i]).toFixed(1)}%`;
    }
  }

  private topK(values: Float32Array, k: number):
      {values: Float32Array, indices: Int32Array} {
    const valuesAndIndices: Array<{value: number, index: number}> = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });

    const topkValues = new Float32Array(k);
    const topkIndices = new Int32Array(k);
    for (let i = 0; i < k; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
    return {values: topkValues, indices: topkIndices};
  }
}

document.registerElement(
    NDArrayLogitsVisualizer.prototype.is, NDArrayLogitsVisualizer);
