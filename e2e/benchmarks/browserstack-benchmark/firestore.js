/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
require('firebase/firestore');
require('firebase/auth');
const firebase = require('firebase/app');
const firebaseConfig = {
  apiKey: process.env.FIREBASE_KEY,
  authDomain: 'learnjs-174218.firebaseapp.com',
  databaseURL: 'https://learnjs-174218.firebaseio.com',
  projectId: 'learnjs-174218',
  storageBucket: 'learnjs-174218.appspot.com',
  messagingSenderId: '834911136599',
  appId: '1:834911136599:web:4b65685455bdf916a1ec12'
};

firebase.initializeApp(firebaseConfig);

firebase.auth()
    .signInAnonymously()
    .then(() => {console.log('Signed into Firebase with anonymous account.')})
    .catch((error) => {
      let errorCode = error.code;
      let errorMessage = error.message;
      console.log(`Error code: ${errorCode}`);
      console.log(`Error message: ${errorMessage}`);
    });


// Reference to the "BenchmarkResults" collection on firestore that contains the
// benchmark results.
const db = firebase.firestore().collection('BenchmarkResults');

/**
 * After being returned from Browserstack, benchmark results are stored as
 * a list of fulfilled promises.
 *
 * As results are being iterated through, this function handles taking a result,
 * serializing it, and pushing it to Firestore.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function addResultToFirestore(resultValue) {
  const firestoreMap =
      formatForFirestore(resultValue, serializeTensors, getReadableDate);

  db.add({result: firestoreMap}).then((ref) => {
    console.log(`Added document to Firestore with ID: ${ref.id}`);
  });
}

/**
 * This functions calls other formatting functions on a benchmark result so that
 * every Firestore entry is compatable and contains desired information
 *
 * @param result Individual result in a list of fulfilled promises
 */
function formatForFirestore(
    resultValue, makeCompatable = serializeTensors, getDate = getReadableDate) {
  let firestoreMap = {};
  firestoreMap.benchmarkInfo = makeCompatable(resultValue);
  firestoreMap.date = getDate();

  return firestoreMap;
}

/**
 *Benchmark results contain tensors that are represented as nested arrays.
 *Nested arrays are not supported on Firestore, so they are serialized
 *before they are stored.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function serializeTensors(resultValue) {
  let kernels = resultValue.memoryInfo.kernels;
  for (kernel of kernels) {
    kernel.inputShapes = JSON.stringify(kernel.inputShapes);
    kernel.outputShapes = JSON.stringify(kernel.outputShapes);
  }
  return resultValue;
}

/**
 * Returns a human readable date so each benchmark has an associated date.
 * Gets date in ISO format so that it is compatible with internal visualisation
 * tool.
 */
function getReadableDate() {
  const fullISODateString = new Date().toISOString();
  const dateOnly = fullISODateString.split('T')[0];
  return dateOnly;
}

exports.addResultToFirestore = addResultToFirestore;
exports.serializeTensors = serializeTensors;
exports.getReadableDate = getReadableDate;
exports.formatForFirestore = formatForFirestore;
exports.db = db;
