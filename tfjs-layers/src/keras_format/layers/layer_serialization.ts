/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {inputLayerClassNames, InputLayerSerialization} from '../input_config';
import {advancedActivationLayerClassNames, AdvancedActivationLayerSerialization} from './advanced_activation_serialization';
import {convolutionalDepthwiseLayerClassNames, ConvolutionalDepthwiseLayerSerialization} from './convolutional_depthwise_serialization';
import {convolutionalLayerClassNames, ConvolutionalLayerSerialization} from './convolutional_serialization';
import {coreLayerClassNames, CoreLayerSerialization} from './core_serialization';
import {embeddingLayerClassNames, EmbeddingLayerSerialization} from './embeddings_serialization';
import {mergeLayerClassNames, MergeLayerSerialization} from './merge_serialization';
import {normalizationLayerClassNames, NormalizationLayerSerialization} from './normalization_serialization';
import {paddingLayerClassNames, PaddingLayerSerialization} from './padding_serialization';
import {poolingLayerClassNames, PoolingLayerSerialization} from './pooling_serialization';
import {recurrentLayerClassNames, RecurrentLayerSerialization} from './recurrent_serialization';

export type LayerSerialization = AdvancedActivationLayerSerialization|
    ConvolutionalDepthwiseLayerSerialization|ConvolutionalLayerSerialization|
    CoreLayerSerialization|EmbeddingLayerSerialization|MergeLayerSerialization|
    NormalizationLayerSerialization|PaddingLayerSerialization|
    PoolingLayerSerialization|RecurrentLayerSerialization|
    InputLayerSerialization;

export type LayerClassName = LayerSerialization['class_name'];

/**
 * A string array of valid Layer class names.
 *
 * This is guaranteed to match the `LayerClassName` union type.
 */
export const layerClassNames: LayerClassName[] = [
  ...advancedActivationLayerClassNames,
  ...convolutionalDepthwiseLayerClassNames, ...convolutionalLayerClassNames,
  ...coreLayerClassNames, ...embeddingLayerClassNames, ...mergeLayerClassNames,
  ...normalizationLayerClassNames, ...paddingLayerClassNames,
  ...poolingLayerClassNames, ...recurrentLayerClassNames,
  ...inputLayerClassNames
];
