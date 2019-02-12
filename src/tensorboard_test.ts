/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {scalar} from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';
import * as tfn from './index';

// tslint:disable-next-line:no-require-imports
const shelljs = require('shelljs');
// tslint:disable-next-line:no-require-imports
const tmp = require('tmp');

describe('tensorboard', () => {
  let tmpLogDir: string;

  beforeEach(() => {
    tmpLogDir = tmp.dirSync().name;
  });

  afterEach(() => {
    if (tmpLogDir != null) {
      shelljs.rm('-rf', tmpLogDir);
    }
  });

  it('Create summaryFileWriter and write scalar', () => {
    const writer = tfn.node.summaryFileWriter(tmpLogDir);
    writer.scalar('foo', 42, 0);
    writer.flush();

    // Currently, we only verify that the file exists and the size
    // increases in a sensible way as we write more scalars to it.
    // The difficulty is in reading the protobuf contents of the event
    // file in JavaScript/TypeScript.
    const fileNames = fs.readdirSync(tmpLogDir);
    expect(fileNames.length).toEqual(1);
    const eventFilePath = path.join(tmpLogDir, fileNames[0]);
    const fileSize0 = fs.statSync(eventFilePath).size;

    writer.scalar('foo', 43, 1);
    writer.flush();
    const fileSize1 = fs.statSync(eventFilePath).size;
    const incrementPerScalar = fileSize1 - fileSize0;
    expect(incrementPerScalar).toBeGreaterThan(0);

    writer.scalar('foo', 44, 2);
    writer.scalar('foo', 45, 3);
    writer.flush();
    const fileSize2 = fs.statSync(eventFilePath).size;
    expect(fileSize2 - fileSize1).toEqual(2 * incrementPerScalar);
  });

  it('Writing tf.Scalar works', () => {
    const writer = tfn.node.summaryFileWriter(tmpLogDir);
    writer.scalar('foo', scalar(42), 0);
    writer.flush();

    // Currently, we only verify that the file exists and the size
    // increases in a sensible way as we write more scalars to it.
    // The difficulty is in reading the protobuf contents of the event
    // file in JavaScript/TypeScript.
    const fileNames = fs.readdirSync(tmpLogDir);
    expect(fileNames.length).toEqual(1);
  });

  it('No crosstalk between two summary writers', () => {
    const logDir1 = path.join(tmpLogDir, '1');
    const writer1 = tfn.node.summaryFileWriter(logDir1);
    writer1.scalar('foo', 42, 0);
    writer1.flush();

    const logDir2 = path.join(tmpLogDir, '2');
    const writer2 = tfn.node.summaryFileWriter(logDir2);
    writer2.scalar('foo', 1.337, 0);
    writer2.flush();

    // Currently, we only verify that the file exists and the size
    // increases in a sensible way as we write more scalars to it.
    // The difficulty is in reading the protobuf contents of the event
    // file in JavaScript/TypeScript.
    let fileNames = fs.readdirSync(logDir1);
    expect(fileNames.length).toEqual(1);
    const eventFilePath1 = path.join(logDir1, fileNames[0]);
    const fileSize1Num0 = fs.statSync(eventFilePath1).size;

    fileNames = fs.readdirSync(logDir2);
    expect(fileNames.length).toEqual(1);
    const eventFilePath2 = path.join(logDir2, fileNames[0]);
    const fileSize2Num0 = fs.statSync(eventFilePath2).size;
    expect(fileSize2Num0).toBeGreaterThan(0);

    writer1.scalar('foo', 43, 1);
    writer1.flush();
    const fileSize1Num1 = fs.statSync(eventFilePath1).size;
    const incrementPerScalar = fileSize1Num1 - fileSize1Num0;
    expect(incrementPerScalar).toBeGreaterThan(0);

    writer1.scalar('foo', 44, 2);
    writer1.scalar('foo', 45, 3);
    writer1.flush();
    const fileSize1Num2 = fs.statSync(eventFilePath1).size;
    expect(fileSize1Num2 - fileSize1Num1).toEqual(2 * incrementPerScalar);

    const fileSize2Num1 = fs.statSync(eventFilePath2).size;
    expect(fileSize2Num1).toEqual(fileSize2Num0);

    writer2.scalar('foo', 1.336, 1);
    writer2.scalar('foo', 1.335, 2);
    writer2.flush();

    const fileSize1Num3 = fs.statSync(eventFilePath1).size;
    expect(fileSize1Num3).toEqual(fileSize1Num2);
    const fileSize2Num2 = fs.statSync(eventFilePath2).size;
    expect(fileSize2Num2 - fileSize2Num1).toEqual(2 * incrementPerScalar);
  });

  it('Writing into existing directory works', () => {
    shelljs.mkdir('-p', tmpLogDir);
    const writer = tfn.node.summaryFileWriter(path.join(tmpLogDir, '22'));
    writer.scalar('foo', 42, 0);
    writer.flush();

    const fileNames = fs.readdirSync(tmpLogDir);
    expect(fileNames.length).toEqual(1);
  });

  it('empty logdir leads to error', () => {
    expect(() => tfn.node.summaryFileWriter('')).toThrowError(/empty string/);
  });
});

describe('tensorBoard callback', () => {
  let tmpLogDir: string;

  beforeEach(() => {
    tmpLogDir = tmp.dirSync().name;
  });

  afterEach(() => {
    if (tmpLogDir != null) {
      shelljs.rm('-rf', tmpLogDir);
    }
  });

  function createModelForTest(): tfn.Model {
    const model = tfn.sequential();
    model.add(
        tfn.layers.dense({units: 5, activation: 'relu', inputShape: [10]}));
    model.add(tfn.layers.dense({units: 1}));
    model.compile(
        {loss: 'meanSquaredError', optimizer: 'sgd', metrics: ['MAE']});
    return model;
  }

  it('fit(): default epoch updateFreq, with validation', async () => {
    const model = createModelForTest();
    const xs = tfn.randomUniform([100, 10]);
    const ys = tfn.randomUniform([100, 1]);
    const valXs = tfn.randomUniform([10, 10]);
    const valYs = tfn.randomUniform([10, 1]);

    // Warm-up training.
    await model.fit(xs, ys, {
      epochs: 1,
      verbose: 0,
      validationData: [valXs, valYs],
      callbacks: tfn.node.tensorBoard(tmpLogDir)
    });

    // Get the initial size of the file.
    // Verify the content of the train and val sub-logdirs.
    const subDirs = fs.readdirSync(tmpLogDir);
    expect(subDirs).toContain('train');
    expect(subDirs).toContain('val');

    const trainLogDir = path.join(tmpLogDir, 'train');
    const trainFiles = fs.readdirSync(trainLogDir);
    const trainFileSize0 =
        fs.statSync(path.join(trainLogDir, trainFiles[0])).size;
    expect(trainFileSize0).toBeGreaterThan(0);
    const valLogDir = path.join(tmpLogDir, 'val');
    const valFiles = fs.readdirSync(valLogDir);
    const valFileSize0 = fs.statSync(path.join(valLogDir, valFiles[0])).size;
    expect(valFileSize0).toBeGreaterThan(0);
    // With updateFreq === epoch, the train and val subset should have generated
    // the same amount of logs.
    expect(valFileSize0).toEqual(trainFileSize0);

    // Actual training run.
    const history = await model.fit(xs, ys, {
      epochs: 3,
      verbose: 0,
      validationData: [valXs, valYs],
      callbacks: tfn.node.tensorBoard(tmpLogDir)
    });
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.val_loss.length).toEqual(3);
    expect(history.history.MAE.length).toEqual(3);
    expect(history.history.val_MAE.length).toEqual(3);

    const trainFileSize1 =
        fs.statSync(path.join(trainLogDir, trainFiles[0])).size;
    const valFileSize1 = fs.statSync(path.join(valLogDir, valFiles[0])).size;
    // We currently only assert that new content has been written to the log
    // file.
    expect(trainFileSize1).toBeGreaterThan(trainFileSize0);
    expect(valFileSize1).toBeGreaterThan(valFileSize0);
    // With updateFreq === epoch, the train and val subset should have generated
    // the same amount of logs.
    expect(valFileSize1).toEqual(trainFileSize1);
  });

  it('fit(): batch updateFreq, with validation', async () => {
    const model = createModelForTest();
    const xs = tfn.randomUniform([100, 10]);
    const ys = tfn.randomUniform([100, 1]);
    const valXs = tfn.randomUniform([10, 10]);
    const valYs = tfn.randomUniform([10, 1]);

    // Warm-up training.
    await model.fit(xs, ys, {
      epochs: 1,
      verbose: 0,
      validationData: [valXs, valYs],
      // Use batch updateFreq here.
      callbacks: tfn.node.tensorBoard(tmpLogDir, {updateFreq: 'batch'})
    });

    // Get the initial size of the file.
    // Verify the content of the train and val sub-logdirs.
    const subDirs = fs.readdirSync(tmpLogDir);
    expect(subDirs).toContain('train');
    expect(subDirs).toContain('val');

    const trainLogDir = path.join(tmpLogDir, 'train');
    const trainFiles = fs.readdirSync(trainLogDir);
    const trainFileSize0 =
        fs.statSync(path.join(trainLogDir, trainFiles[0])).size;
    expect(trainFileSize0).toBeGreaterThan(0);
    const valLogDir = path.join(tmpLogDir, 'val');
    const valFiles = fs.readdirSync(valLogDir);
    const valFileSize0 = fs.statSync(path.join(valLogDir, valFiles[0])).size;
    expect(valFileSize0).toBeGreaterThan(0);
    // The train subset should have generated more logs than the val subset,
    // because the train subset gets logged every batch, while the val subset
    // gets logged every epoch.
    expect(trainFileSize0).toBeGreaterThan(valFileSize0);

    // Actual training run.
    const history = await model.fit(xs, ys, {
      epochs: 3,
      verbose: 0,
      validationData: [valXs, valYs],
      callbacks: tfn.node.tensorBoard(tmpLogDir)
    });
    expect(history.history.loss.length).toEqual(3);
    expect(history.history.val_loss.length).toEqual(3);
    expect(history.history.MAE.length).toEqual(3);
    expect(history.history.val_MAE.length).toEqual(3);

    const trainFileSize1 =
        fs.statSync(path.join(trainLogDir, trainFiles[0])).size;
    const valFileSize1 = fs.statSync(path.join(valLogDir, valFiles[0])).size;
    // We currently only assert that new content has been written to the log
    // file.
    expect(trainFileSize1).toBeGreaterThan(trainFileSize0);
    expect(valFileSize1).toBeGreaterThan(valFileSize0);
    // The train subset should have generated more logs than the val subset,
    // because the train subset gets logged every batch, while the val subset
    // gets logged every epoch.
    expect(trainFileSize1).toBeGreaterThan(valFileSize1);
  });

  it('Invalid updateFreq value causes error', async () => {
    expect(() => tfn.node.tensorBoard(tmpLogDir, {
      // tslint:disable-next-line:no-any
      updateFreq: 'foo' as any
    })).toThrowError(/Expected updateFreq/);
  });
});
