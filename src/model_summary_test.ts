/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// Unit tests for tf.LayersModel.summary() and tf.Sequential.summary().

import * as tfl from './index';
import {describeMathCPU} from './utils/test_utils';

function getRandomLayerOrModelName(length = 12) {
  return 'L' + Math.random().toFixed(length - 1).slice(2);
}

describeMathCPU('LayersModel.summary', () => {
  let consoleLogHistory: string[];

  beforeEach(() => {
    consoleLogHistory = [];
    spyOn(console, 'log').and.callFake((message?: string) => {
      consoleLogHistory.push(message);
    });
  });

  afterEach(() => {
    consoleLogHistory = [];
  });

  it('Sequential model: one layer', () => {
    const layerName = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${layerName} (Dense)         [null,3]                  33        `,
      '=================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);
  });

  it('Sequential model: one layer: custom lineLength', () => {
    const layerName = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    const lineLength = 70;
    model.summary(lineLength);
    expect(consoleLogHistory).toEqual([
      '______________________________________________________________________',
      'Layer (type)                   Output shape                Param #    ',
      '======================================================================',
      `${layerName} (Dense)           [null,3]                    33         `,
      '======================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
  });

  it('Sequential model: one layer: custom positions', () => {
    const layerName = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });
    const lineLength = 70;
    const positions: number[] = [0.5, 0.8, 1.0];
    model.summary(lineLength, positions);
    expect(consoleLogHistory).toEqual([
      '______________________________________________________________________',
      'Layer (type)                       Output shape         Param #       ',
      '======================================================================',
      `${layerName} (Dense)               [null,3]             33            `,
      '======================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
  });

  it('Sequential model: one layer: custom printFn', () => {
    const layerName = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10], name: layerName})]
    });

    const messages: string[] = [];
    // tslint:disable-next-line:no-any
    function rerouteLog(message?: any, ...optionalParams: any[]) {
      messages.push(message);
    }

    model.summary(null, null, rerouteLog);
    expect(messages).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${layerName} (Dense)         [null,3]                  33        `,
      '=================================================================',
      'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);

    // console.log should have received no calls.
    expect(consoleLogHistory).toEqual([]);
  });

  it('Sequential model: three layers', () => {
    const lyrName01 = getRandomLayerOrModelName();
    const lyrName02 = getRandomLayerOrModelName();
    const lyrName03 = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName01}),
        tfl.layers.dense({units: 3, name: lyrName02}),
        tfl.layers.dense({units: 1, name: lyrName03}),
      ]
    });
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
  });

  it('Sequential model: with non-trainable layers', () => {
    const lyrName01 = getRandomLayerOrModelName();
    const lyrName02 = getRandomLayerOrModelName();
    const lyrName03 = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [
        tfl.layers.flatten({inputShape: [2, 5], name: lyrName01}),
        tfl.layers.dense({units: 3, name: lyrName02, trainable: false}),
        tfl.layers.dense({units: 1, name: lyrName03}),
      ]
    });
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 4',
      'Non-trainable params: 33',
      '_________________________________________________________________',
    ]);
    consoleLogHistory = [];

    // Setting the entire model to non-trainable should be reflected
    // in the summary.
    model.trainable = false;
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 0',
      'Non-trainable params: 37',
      '_________________________________________________________________',
    ]);
    consoleLogHistory = [];

    // Setting the model's trainable property should be reflected in the
    // new summary. But the initially untrainable layer should still stay
    // untrainable.
    model.trainable = true;
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Flatten)       [null,10]                 0         `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName03} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 4',
      'Non-trainable params: 33',
      '_________________________________________________________________',
    ]);
    consoleLogHistory = [];
  });

  it('Sequential model with Embedding layer', () => {
    const lyrName01 = getRandomLayerOrModelName();
    const lyrName02 = getRandomLayerOrModelName();
    const model = tfl.sequential({
      layers: [
        tfl.layers.embedding({
          inputDim: 10,
          outputDim: 8,
          inputShape: [null, 5],
          name: lyrName01
        }),
        tfl.layers.dense({units: 3, name: lyrName02}),
      ]
    });
    model.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${lyrName01} (Embedding)     [null,null,5,8]           80        `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,null,5,3]           27        `,
      '=================================================================',
      'Total params: 107', 'Trainable params: 107', 'Non-trainable params: 0',
      '_________________________________________________________________'
    ]);
  });

  it('Sequential model: nested', () => {
    const mdlName01 = getRandomLayerOrModelName();
    const innerModel = tfl.sequential({
      layers: [tfl.layers.dense({units: 3, inputShape: [10]})],
      name: mdlName01
    });
    const outerModel = tfl.sequential();
    outerModel.add(innerModel);

    const lyrName02 = getRandomLayerOrModelName();
    outerModel.add(tfl.layers.dense({units: 1, name: lyrName02}));

    outerModel.summary();
    expect(consoleLogHistory).toEqual([
      '_________________________________________________________________',
      'Layer (type)                 Output shape              Param #   ',
      '=================================================================',
      `${mdlName01} (Sequential)    [null,3]                  33        `,
      '_________________________________________________________________',
      `${lyrName02} (Dense)         [null,1]                  4         `,
      '=================================================================',
      'Total params: 37',
      'Trainable params: 37',
      'Non-trainable params: 0',
      '_________________________________________________________________',
    ]);
  });

  it('Functional model', () => {
    const lyrName01 = getRandomLayerOrModelName();
    const input1 = tfl.input({shape: [3], name: lyrName01});
    const lyrName02 = getRandomLayerOrModelName();
    const input2 = tfl.input({shape: [4], name: lyrName02});
    const lyrName03 = getRandomLayerOrModelName();
    const input3 = tfl.input({shape: [5], name: lyrName03});
    const lyrName04 = getRandomLayerOrModelName();
    const concat1 =
        tfl.layers.concatenate({name: lyrName04}).apply([input1, input2]) as
        tfl.SymbolicTensor;
    const lyrName05 = getRandomLayerOrModelName();
    const output =
        tfl.layers.concatenate({name: lyrName05}).apply([concat1, input3]) as
        tfl.SymbolicTensor;
    const model =
        tfl.model({inputs: [input1, input2, input3], outputs: output});

    const lineLength = 70;
    const positions: number[] = [0.42, 0.64, 0.75, 1];
    model.summary(lineLength, positions);
    expect(consoleLogHistory).toEqual([
      '______________________________________________________________________',
      'Layer (type)                 Output shape   Param # Receives inputs   ',
      '======================================================================',
      `${lyrName01} (InputLayer)    [null,3]       0                         `,
      '______________________________________________________________________',
      `${lyrName02} (InputLayer)    [null,4]       0                         `,
      '______________________________________________________________________',
      `${lyrName04} (Concatenate)   [null,7]       0       ${lyrName01}[0][0]`,
      `                                                    ${lyrName02}[0][0]`,
      '______________________________________________________________________',
      `${lyrName03} (InputLayer)    [null,5]       0                         `,
      '______________________________________________________________________',
      `${lyrName05} (Concatenate)   [null,12]      0       ${lyrName04}[0][0]`,
      `                                                    ${lyrName03}[0][0]`,
      '======================================================================',
      'Total params: 0', 'Trainable params: 0', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
  });

  it('LayersModel with multiple outputs', () => {
    const lyrName01 = getRandomLayerOrModelName();
    const input1 = tfl.input({shape: [3, 4], name: lyrName01});
    const lyrName02 = getRandomLayerOrModelName();
    const outputs =
        tfl.layers.simpleRNN({units: 2, returnState: true, name: lyrName02})
            .apply(input1) as tfl.SymbolicTensor[];
    const model = tfl.model({inputs: input1, outputs});
    const lineLength = 70;
    model.summary(lineLength);
    expect(consoleLogHistory).toEqual([
      '______________________________________________________________________',
      'Layer (type)                   Output shape                Param #    ',
      '======================================================================',
      `${lyrName01} (InputLayer)      [null,3,4]                  0          `,
      '______________________________________________________________________',
      `${lyrName02} (SimpleRNN)       [[null,2],[null,2]]         14         `,
      '======================================================================',
      'Total params: 14', 'Trainable params: 14', 'Non-trainable params: 0',
      '______________________________________________________________________'
    ]);
  });
});
