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

import firebase from 'firebase/app';
import {EnvironmentInfo, Task, TaskLog, TaskType, VersionSet} from './types';

let firebaseInitialized = false;
async function initFirebase() {
  if (!firebaseInitialized) {
    await firebase.initializeApp({
      authDomain: 'jstensorflow.firebaseapp.com',
      projectId: 'jstensorflow'
    });
    firebaseInitialized = true;
  }
}

export async function addTaskLogsToFirestore(taskLogs: TaskLog[]) {
  await initFirebase();

  const db = firebase.firestore();
  const batch = db.batch();
  const collection = db.collection('TaskLogs');

  for (let i = 0; i < taskLogs.length; ++i) {
    const taskLog = taskLogs[i];
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
  await initFirebase();

  const db = firebase.firestore();
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
 await initFirebase();

 const db = firebase.firestore();
 const collection = db.collection('VersionSets');
 const doc = await collection.add(versionSet);
 return doc.id;
}

/**
 *
 */
export async function addOrGetTaskId(
    taskType: TaskType, taskName: string, functionName?: string): Promise<string> {
  await initFirebase();

  const db = firebase.firestore();
  const collection = db.collection('Tasks');
  const query = collection
      .where('taskType', '==', taskType)
      .where('taskName', '==', taskName)
      .where('functionName', '==', functionName);
  const result = await query.get();
  let taskId: string;
  result.forEach(row => {
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
