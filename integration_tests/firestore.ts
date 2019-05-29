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

import * as firebase from 'firebase/app';
// tslint:disable-next-line:max-line-length
import {EnvironmentInfo, Task, TaskType, VersionSet, BenchmarkRun} from './types';

// tslint:disable-next-line:no-any
declare let __karma__: any;

/** Determine whether this file is running in Node.js. */
export function inNodeJS(): boolean {
  // Note this is not a generic way of testing if we are in Node.js.
  // The logic here is specific to the scripts in this folder.
  return typeof module !== 'undefined' && typeof process !== 'undefined' &&
      typeof __karma__ === 'undefined';
}

let firebaseInitialized = false;
// let firestore: firebase.firestore.Firestore;
async function initFirebase(): Promise<firebase.firestore.Firestore> {
  if (!inNodeJS()) {
    if (!firebaseInitialized) {
      await firebase.initializeApp({
        authDomain: 'jstensorflow.firebaseapp.com',
        projectId: 'jstensorflow'
      });
      firebaseInitialized = true;
      // firestore = firebase.firestore();
    }
    return firebase.firestore();
  } else {
    // In Node.js.
    // TODO(cais): Find a way to get rid of the hacky-looking import while
    // avoiding code duplication between Node.js and browser. Currently, this
    // string splitting helps us avoid an error during compilation of grpc,
    // which is a dependenc of firebase-admin. The compilation error happens
    // only in the browser, but not in Node.js.
    // For context: Firebase Firestore has two different client libraries
    // for Node.js and browser.
    // tslint:disable-next-line:no-require-imports
    const admin = require('firebase-' + 'admin');
    if (!firebaseInitialized) {
      admin.initializeApp({
        credential: admin.credential.applicationDefault()
      });
      firebaseInitialized = true;
    }
    return admin.firestore();
    // throw new Error('Not implemented');
  }
}

/**
 * Add BenchmarkRun data to Firestore.
 *
 * @param run An array of BenchmarkRun, i.e., the results from a number
 *   of benchmark tasks.
 */
export async function addBenchmarkRunsToFirestore(run: BenchmarkRun[]) {
  const db = await initFirebase();
  const batch = db.batch();
  const collection = db.collection('BenchmarkRuns');

  for (let i = 0; i < run.length; ++i) {
    const taskLog = run[i];
    const ref = collection.doc();
    batch.set(ref, taskLog);
  }

  await batch.commit();
}

/**
 * Add an entry of EnvironmentInfo to Firestore.
 *
 * @param environmentInfo Environment information to be added.
 * @return ID of the newly created doc.
 */
export async function addEnvironmentInfoToFirestore(
     environmentInfo: EnvironmentInfo): Promise<string> {
  const db = await initFirebase();
  const collection = db.collection('Environments');
  const doc = await collection.add(environmentInfo);
  return doc.id;
}

/**
 * Add an entry of EnvironmentInfo to Firestore.
 *
 * @param environmentInfo Environment information to be added.
 * @return ID of the newly created doc.
 */
export async function addVersionSetToFirestore(
    versionSet: VersionSet): Promise<string> {
  const db = await initFirebase();
  const collection = db.collection('VersionSets');
  const doc = await collection.add(versionSet);
  return doc.id;
}

/**
 * Add a task if it doesn't exist in the TaskCollection.
 *
 * @param taskType Task type
 * @param taskName Task name. For a model task, this is the model's name.
 * @param functionName Function name. For a model task, options are predict,
 *   fit and fitDataset().
 * @return Task ID from TaskCollection.
 */
export async function addOrGetTaskId(
    taskType: TaskType, taskName: string, functionName?: string):
    Promise<string> {
  const db = await initFirebase();
  const collection = db.collection('Tasks');
  const query = collection
      .where('taskType', '==', taskType)
      .where('taskName', '==', taskName)
      .where('functionName', '==', functionName);
  const result = await query.get();
  let taskId: string;
  // tslint:disable-next-line:no-any
  result.forEach((row: any) => {
    taskId = row.id;
    return;
  });

  if (taskId == null) {
    const doc = await collection.add({
      taskType,
      taskName,
      functionName
    } as Task);
    taskId = doc.id;
  }
  return taskId;
}
