/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

const firebase = require("firebase/app");
const firebaseConfig = {
  apiKey: process.env.FIREBASE_KEY,
  authDomain: "learnjs-174218.firebaseapp.com",
  databaseURL: "https://learnjs-174218.firebaseio.com",
  projectId: "learnjs-174218",
  storageBucket: "learnjs-174218.appspot.com",
  messagingSenderId: "834911136599",
  appId: "1:834911136599:web:4b65685455bdf916a1ec12"
};
require("firebase/firestore");
firebase.initializeApp(firebaseConfig);

/**
*Reference to the "posts" collection on firestore that contains benchmark
*results.
*/
const db = firebase.firestore().collection("posts");

/**
 * After being returned from Browserstack, benchmark results are stored as
 * a list of fulfilled promises.
 *
 * As results are being iterated through, this function handles taking a result,
 * serializing it, and pushing it to Firestore.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function addResultToFirestore(result) {
  let firestoreMap = {};
  firestoreMap.benchmarkInfo = serializeTensors(result).value;

  db.add({
    result: firestoreMap
  })
  .then((ref) => {
    console.log("Added document to Firebase with ID: ", ref.id);
  });
}

/**
 *Benchmark results contain tensors that are represented as nested arrays.
 *Nested arrays are not supported on Firestore, so they are serialized
 *before they are stored.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function serializeTensors(result) {
  let kernels = result.value.memoryInfo.kernels;
  for (kernel of kernels) {
    kernel.inputShapes = JSON.stringify(kernel.inputShapes);
    kernel.outputShapes = JSON.stringify(kernel.outputShapes);
  }
  return result
}

exports.addResultToFirestore = addResultToFirestore;
