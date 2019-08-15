/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {io, zeros} from '@tensorflow/tfjs-core';

import * as tfl from './index';
import {Sequential} from './models';
import {plainObjectCheck, MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH} from './user_defined_metadata';

describe('plainObjectCheck', () => {
  it('Primitives', () => {
    // `undefined` is not valid JSON.
    expect(plainObjectCheck(undefined)).toEqual(false);
    // `null` is valid JSON.
    expect(plainObjectCheck(null)).toEqual(true);
    expect(plainObjectCheck(true)).toEqual(true);
    expect(plainObjectCheck(1337)).toEqual(true);
    expect(plainObjectCheck('foo')).toEqual(true);
  });
  it('Complex objects lead to false', () => {
    expect(plainObjectCheck(new Date())).toEqual(false);
    expect(plainObjectCheck(new Float32Array([1, 2])))
        .toEqual(false);
    expect(plainObjectCheck(new ArrayBuffer(4))).toEqual(false);
    expect(plainObjectCheck(new Error())).toEqual(false);
    expect(plainObjectCheck(zeros([2, 3]))).toEqual(false);
  });
  it('POJOs lead to true', () => {
    expect(plainObjectCheck({})).toEqual(true);
    expect(plainObjectCheck({
      'key1': 'foo',
      'key2': 1337,
      'key3': false
    })).toEqual(true);
    expect(plainObjectCheck({
      'key1': {
        'key1_1': [1, 3, 3, 7, [42]],
        'key1_2': null,
        'key1_3': ['foo', 'bar', null, {}, {'qux': 233}],
      },
      'key2': 1337,
      'key3': false
    })).toEqual(true);
  });
  it('POJOs with invalid value types lead to false', () => {
    expect(plainObjectCheck({
      'key1': new Date(),
      'key2': 1337
    })).toEqual(false);
    expect(plainObjectCheck({
      'key1': new ArrayBuffer(3),
      'key2': 1337
    })).toEqual(false);
    expect(plainObjectCheck({
      'key1': 'foo',
      'key2': undefined
    })).toEqual(false);
    expect(plainObjectCheck({
      'key1': 'foo',
      'key2': {
        'tensor': zeros([2, 3])
      }
    })).toEqual(false);
  });
  it('Arrays of POJO lead to true', () => {
    expect(plainObjectCheck([])).toEqual(true);
    expect(plainObjectCheck([{}, {}])).toEqual(true);
    expect(plainObjectCheck([{
      'key1': 'foo',
      'key2': 1337,
      'key3': false
    }])).toEqual(true);
    expect(plainObjectCheck([{
      'key1': {
        'key1_1': [1, 3, 3, 7],
        'key1_2': null,
        'key1_3': ['foo', 'bar', null, {}, {'qux': 233}],
      },
      'key2': 1337,
      'key3': false
    }])).toEqual(true);
  });
});

describe('Save and load model with metadata', () => {

  function createSequentialModelForTest(): Sequential {
    const model = tfl.sequential();
    model.add(tfl.layers.dense({
      units: 3,
      inputShape: [10],
      activation: 'softmax'
    }));
    return model;
  }

  function createFunctionalModelForTest(): tfl.LayersModel {
    const input1 = tfl.input({shape: [3]});
    const input2 = tfl.input({shape: [4]});
    const dense1 = tfl.layers.dense({units: 2, inputShape: [3]});
    const dense2 = tfl.layers.dense({units: 1, inputShape: [4]});
    const y1 = dense1.apply(input1) as tfl.SymbolicTensor;
    const y2 = dense2.apply(input2) as tfl.SymbolicTensor;
    const output = tfl.layers.concatenate().apply([y1, y2]) as
        tfl.SymbolicTensor;
    return tfl.model({inputs: [input1, input2], outputs: output});
  }

  for (const modelType of ['sequential', 'functional']) {
    it(`Valid user-defined metadata round trip: ${modelType}`, async () => {
      const model = modelType === 'sequential' ?
          createSequentialModelForTest() : createFunctionalModelForTest();
      const userDefinedMetadata = {'outputLabels': ['foo', 'bar', 'baz']};
      model.setUserDefinedMetadata(userDefinedMetadata);
      expect(model.getUserDefinedMetadata()).toEqual(userDefinedMetadata);
      let savedArtifacts: io.ModelArtifacts;
      await model.save(
          io.withSaveHandler(async (artifacts: io.ModelArtifacts) => {
            savedArtifacts = artifacts;
            return {modelArtifactsInfo: null};
          }));
      const reloadedModel = await tfl.loadLayersModel(
          io.fromMemory(savedArtifacts));
      expect(reloadedModel.getUserDefinedMetadata())
          .toEqual(userDefinedMetadata);
    });
  }
  for (const modelType of ['sequential', 'functional']) {
    it(`Invalid user metadata leads to error: ${modelType}`, async () => {
      const model = modelType === 'sequential' ?
          createSequentialModelForTest() : createFunctionalModelForTest();
      expect(() => model.setUserDefinedMetadata(
          JSON.stringify({'outputLabels': ['foo', 'bar', 'baz']})))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata(
          ['foo', 'bar', 'baz']))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata(
          {'foo': zeros([2, 3]), 'outputLabels': ['foo', 'bar', 'baz']}))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata(undefined))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata(null))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata('foo'))
          .toThrowError(/is expected to be a JSON object, but is not/);
      expect(() => model.setUserDefinedMetadata(1337))
          .toThrowError(/is expected to be a JSON object, but is not/);
    });
  }
  for (const modelType of ['sequential', 'functional']) {
    it(`Large metadata size leads to warning: ${modelType}`, async () => {
      const warningMessages: string[] = [];
      spyOn(console, 'warn').and.callFake((message: string) => {
        warningMessages.push(message);
      });
      const largeMetadata: {} = {
        'metadata': 'x'.repeat(MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH)
      };
      const model = modelType === 'sequential' ?
          createSequentialModelForTest() : createFunctionalModelForTest();
      model.setUserDefinedMetadata(largeMetadata);
      await model.save(
          io.withSaveHandler(async (artifacts: io.ModelArtifacts) => {
            return {modelArtifactsInfo: null};
          }));
      expect(warningMessages.length).toEqual(1);
      expect(warningMessages).toMatch(/is too large in size/);
    });
  }
  it('MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH value', () => {
    expect(MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH).toBeGreaterThan(0);
    expect(Number.isInteger(MAX_USER_DEFINED_METADATA_SERIALIZED_LENGTH))
        .toEqual(true);

  });
});
