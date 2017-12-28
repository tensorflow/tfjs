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

import '../test_env';
import {CheckpointLoader, CheckpointManifest} from './checkpoint_loader';

describe('Checkpoint var loader', () => {
  let xhrObj: XMLHttpRequest;

  beforeEach(() => {
    xhrObj = jasmine.createSpyObj(
        'xhrObj', ['addEventListener', 'open', 'send', 'onload', 'onerror']);
    // tslint:disable-next-line:no-any
    spyOn(window as any, 'XMLHttpRequest').and.returnValue(xhrObj);
  });

  it('Load manifest and a variable', (doneFn) => {
    const fakeCheckpointManifest: CheckpointManifest = {
      'fakeVar1': {filename: 'fakeFile1', shape: [10]},
      'fakeVar2': {filename: 'fakeFile2', shape: [5, 5]}
    };

    const varLoader = new CheckpointLoader('fakeModel');
    varLoader.getCheckpointManifest().then(checkpoint => {
      expect(checkpoint).toEqual(fakeCheckpointManifest);

      const buffer =
          new ArrayBuffer(4 * fakeCheckpointManifest['fakeVar1'].shape[0]);
      const view = new Float32Array(buffer);
      for (let i = 0; i < 10; i++) {
        view[i] = i;
      }

      varLoader.getVariable('fakeVar1').then(ndarray => {
        expect(ndarray.shape).toEqual(fakeCheckpointManifest['fakeVar1'].shape);
        expect(ndarray.dataSync()).toEqual(view);
        doneFn();
      });
      // tslint:disable-next-line:no-any
      (xhrObj as any).response = buffer;
      // tslint:disable-next-line:no-any
      (xhrObj as any).onload();
    });
    // tslint:disable-next-line:no-any
    (xhrObj as any).responseText = JSON.stringify(fakeCheckpointManifest);
    // tslint:disable-next-line:no-any
    (xhrObj as any).onload();
  });

  it('Load manifest error', () => {
    const varLoader = new CheckpointLoader('fakeModel');
    varLoader.getCheckpointManifest();
    // tslint:disable-next-line:no-any
    expect(() => (xhrObj as any).onerror()).toThrowError();
  });

  it('Load non-existent variable throws error', (doneFn) => {
    const fakeCheckpointManifest:
        CheckpointManifest = {'fakeVar1': {filename: 'fakeFile1', shape: [10]}};

    const varLoader = new CheckpointLoader('fakeModel');
    varLoader.getCheckpointManifest().then(checkpoint => {
      expect(() => varLoader.getVariable('varDoesntExist')).toThrowError();
      doneFn();
    });
    // tslint:disable-next-line:no-any
    (xhrObj as any).responseText = JSON.stringify(fakeCheckpointManifest);
    // tslint:disable-next-line:no-any
    (xhrObj as any).onload();
  });

  it('Load variable throws error', (doneFn) => {
    const fakeCheckpointManifest:
        CheckpointManifest = {'fakeVar1': {filename: 'fakeFile1', shape: [10]}};

    const varLoader = new CheckpointLoader('fakeModel');
    varLoader.getCheckpointManifest().then(checkpoint => {
      varLoader.getVariable('fakeVar1');
      // tslint:disable-next-line:no-any
      expect(() => (xhrObj as any).onerror()).toThrowError();
      doneFn();
    });
    // tslint:disable-next-line:no-any
    (xhrObj as any).responseText = JSON.stringify(fakeCheckpointManifest);
    // tslint:disable-next-line:no-any
    (xhrObj as any).onload();
  });

  it('Load variable but 404 not found', (doneFn) => {
    const fakeCheckpointManifest:
        CheckpointManifest = {'fakeVar1': {filename: 'fakeFile1', shape: [10]}};

    const varLoader = new CheckpointLoader('fakeModel');
    varLoader.getCheckpointManifest().then(checkpoint => {
      varLoader.getVariable('fakeVar1');
      // tslint:disable-next-line:no-any
      expect(() => (xhrObj as any).onload()).toThrowError();
      doneFn();
    });
    // tslint:disable-next-line:no-any
    (xhrObj as any).responseText = JSON.stringify(fakeCheckpointManifest);
    // tslint:disable-next-line:no-any
    (xhrObj as any).status = 404;
    // tslint:disable-next-line:no-any
    (xhrObj as any).onload();
  });
});
