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

import {Graph, Tensor} from 'deeplearn';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

import * as layer_builder from './layer_builder';
import {LayerBuilder, LayerName, LayerWeightsDict} from './layer_builder';
import {ModelBuilder} from './model-builder';
import * as model_builder_util from './model_builder_util';

// tslint:disable-next-line:variable-name
export let ModelLayerPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'model-layer',
  properties: {
    layerName: String,
    inputShapeDisplay: String,
    outputShapeDisplay: String,
    isStatic: {type: Boolean, value: false},
    layerNames: Array,
    selectedLayerName: String,
    hasError: {type: Boolean, value: false},
    errorMessages: Array,
  }
});

export class ModelLayer extends ModelLayerPolymer {
  // Polymer properties.
  inputShapeDisplay: string;
  outputShapeDisplay: string;
  layerNames: LayerName[];
  selectedLayerName: LayerName;
  hasError: boolean;
  errorMessages: string[];

  private modelBuilder: ModelBuilder;
  layerBuilder: LayerBuilder;
  private inputShape: number[];
  private outputShape: number[];

  private paramContainer: HTMLDivElement;

  initialize(modelBuilder: ModelBuilder, inputShape: number[]) {
    this.modelBuilder = modelBuilder;
    this.paramContainer =
        this.querySelector('.param-container') as HTMLDivElement;
    this.layerNames = [
      'Fully connected', 'ReLU', 'Convolution', 'Max pool', 'Reshape', 'Flatten'
    ];
    this.inputShape = inputShape;
    this.buildParamsUI('Fully connected', this.inputShape);

    this.querySelector('.dropdown-content')
        .addEventListener(
            // tslint:disable-next-line:no-any
            'iron-activate', (event: any) => {
              this.buildParamsUI(
                  event.detail.selected as LayerName, this.inputShape);
            });

    this.querySelector('#remove-layer').addEventListener('click', (event) => {
      modelBuilder.removeLayer(this);
    });
  }

  setInputShape(shape: number[]): number[] {
    this.inputShape = shape;
    this.inputShapeDisplay =
        model_builder_util.getDisplayShape(this.inputShape);

    const errors: string[] = [];
    const validationErrors = this.layerBuilder.validate(this.inputShape);
    if (validationErrors != null) {
      for (let i = 0; i < validationErrors.length; i++) {
        errors.push('Error: ' + validationErrors[i]);
      }
    }

    try {
      this.outputShape = this.layerBuilder.getOutputShape(this.inputShape);
    } catch (e) {
      errors.push(e);
    }
    this.outputShapeDisplay =
        model_builder_util.getDisplayShape(this.outputShape);

    if (errors.length > 0) {
      this.hasError = true;
      this.errorMessages = errors;
    } else {
      this.hasError = false;
      this.errorMessages = [];
    }

    return this.outputShape;
  }

  isValid(): boolean {
    return !this.hasError;
  }

  getOutputShape(): number[] {
    return this.outputShape;
  }

  addLayer(
      g: Graph, network: Tensor, index: number,
      weights: LayerWeightsDict|null): Tensor {
    return this.layerBuilder.addLayer(
        g, network, this.inputShape, index, weights);
  }

  /**
   * Build parameters for the UI for a given op type. This is called when the
   * op is added, and when the op type changes.
   */
  buildParamsUI(
      layerName: LayerName, inputShape: number[],
      layerBuilderJson?: LayerBuilder) {
    this.selectedLayerName = layerName;

    this.layerBuilder =
        layer_builder.getLayerBuilder(layerName, layerBuilderJson);

    // Clear any existing parameters.
    this.paramContainer.innerHTML = '';

    // Add all the parameters to the UI.
    const layerParams = this.layerBuilder.getLayerParams();
    for (let i = 0; i < layerParams.length; i++) {
      const initialValue = layerBuilderJson != null ?
          layerParams[i].getValue() :
          layerParams[i].initialValue(inputShape);
      this.addParamField(
          layerParams[i].label, initialValue, layerParams[i].setValue,
          layerParams[i].type, layerParams[i].min, layerParams[i].max);
    }
    this.modelBuilder.layerParamChanged();
  }

  loadParamsFromLayerBuilder(
      inputShape: number[], layerBuilderJson: LayerBuilder) {
    this.buildParamsUI(
        layerBuilderJson.layerName, inputShape, layerBuilderJson);
  }

  private addParamField(
      label: string, initialValue: number|string,
      setValue: (value: number|string) => void, type: 'number'|'text',
      min?: number, max?: number) {
    const input = document.createElement('paper-input');
    input.setAttribute('always-float-label', 'true');
    input.setAttribute('label', label);
    input.setAttribute('value', initialValue.toString());
    input.setAttribute('type', type);
    if (type === 'number') {
      input.setAttribute('min', min.toString());
      input.setAttribute('max', max.toString());
    }
    input.className = 'param-input';
    this.paramContainer.appendChild(input);

    // Update the parent when this changes.
    input.addEventListener('input', (event) => {
      if (type === 'number') {
        // tslint:disable-next-line:no-any
        setValue((event.target as any).valueAsNumber as number);
      } else {
        // tslint:disable-next-line:no-any
        setValue((event.target as any).value as string);
      }
      this.modelBuilder.layerParamChanged();
    });
    setValue(initialValue);
  }
}

document.registerElement(ModelLayer.prototype.is, ModelLayer);
