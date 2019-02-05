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
