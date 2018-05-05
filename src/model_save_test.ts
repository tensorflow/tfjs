/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {io} from '@tensorflow/tfjs-core';

import {Dense} from './layers/core';
import {Sequential} from './models';
import {describeMathCPUAndGPU} from './utils/test_utils';

describeMathCPUAndGPU('Model.save', () => {
  class IOHandlerForTest implements io.IOHandler {
    savedArtifacts: io.ModelArtifacts;

    async save(modelArtifacts: io.ModelArtifacts): Promise<io.SaveResult> {
      this.savedArtifacts = modelArtifacts;
      return {modelArtifactsInfo: null};
    }
  }

  class EmptyIOHandler implements io.IOHandler {}

  it('Saving all weights succeeds', async done => {
    const model = new Sequential();
    model.add(new Dense({units: 3, inputShape: [5]}));
    const handler = new IOHandlerForTest();

    model.save(handler)
        .then(saveResult => {
          expect(handler.savedArtifacts.modelTopology)
              .toEqual(model.toJSON(null, false));
          expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
          expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([5, 3]);
          expect(handler.savedArtifacts.weightSpecs[0].dtype)
              .toEqual('float32');
          expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([3]);
          expect(handler.savedArtifacts.weightSpecs[1].dtype)
              .toEqual('float32');
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Saving only trainable weights succeeds', async done => {
    const model = new Sequential();
    model.add(new Dense({units: 3, inputShape: [5], trainable: false}));
    model.add(new Dense({units: 2}));
    const handler = new IOHandlerForTest();

    model.save(handler, {trainableOnly: true})
        .then(saveResult => {
          expect(handler.savedArtifacts.modelTopology)
              .toEqual(model.toJSON(null, false));
          // Verify that only the trainable weights (i.e., weights from the
          // 2nd, trainable Dense layer) are saved.
          expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
          expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([3, 2]);
          expect(handler.savedArtifacts.weightSpecs[0].dtype)
              .toEqual('float32');
          expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
              .toBeGreaterThan(0);
          expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([2]);
          expect(handler.savedArtifacts.weightSpecs[1].dtype)
              .toEqual('float32');
          done();
        })
        .catch(err => {
          console.error(err.stack);
        });
  });

  it('Saving to a handler without save method fails', async done => {
    const model = new Sequential();
    model.add(new Dense({units: 3, inputShape: [5]}));
    const handler = new EmptyIOHandler();
    model.save(handler)
        .then(saveResult => {
          fail(
              'Saving with an IOHandler without `save` succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message)
              .toEqual(
                  'Model.save() cannot proceed because the IOHandler ' +
                  'provided does not have the `save` attribute defined.');
          done();
        });
  });
});
