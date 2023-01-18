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

const { initializeApp, deleteApp, applicationDefault, cert } = require('firebase-admin/app');
const { getFirestore, Timestamp, FieldValue } = require('firebase-admin/firestore');

let app;
/**
 * Initializes Firebase, signs in with secret credentials, and accesses the
 * Firestore collection of results.
 *
 * @param firebaseConfig A configuration with Firebase credentials
 */
async function runFirestore() {
  try {
    app = initializeApp({
      credential: applicationDefault()
    });
    const db = getFirestore();

    console.log('\nSuccesfuly signed into Firebase.');
    return db.collection('BenchmarkResults');
  } catch (err) {
    console.warn(`Failed to connect to firebase database: ${err.message}`);
    throw err;
  }
}

/**
 * Deletes the Firebase instance, which allows the Node.js process to finish.
 */
async function endFirebaseInstance() {
  await deleteApp(app);
  console.log('Exited Firebase instance.');
}

/**
 * After being returned from Browserstack, benchmark results are stored as
 * a list of fulfilled promises.
 *
 * As results are being iterated through, this function handles taking a result,
 * serializing it, and pushing it to Firestore.
 *
 * @param db Reference to Firestore collection
 * @param resultId ID of value added to Firestore
 * @param result Individual result in a list of fulfilled promises
 */
async function addResultToFirestore(db, resultId, result) {
  try {
    const firestoreMap =
      formatForFirestore(result, makeCompatableWithFirestore, getReadableDate);
    await db.add({ result: firestoreMap }).then((ref) => {
      console.log(`Added ${resultId} to Firestore with ID: ${ref.id}`);
    });
  } catch (err) {
    throw err;
  }
}

/**
 * This functions calls other formatting functions on a benchmark result so that
 * every Firestore entry is compatable and contains desired information
 *
 * @param result Individual result in a list of fulfilled promises
 */
function formatForFirestore(
  result, makeCompatable = makeCompatableWithFirestore,
  getDate = getReadableDate) {
  let firestoreMap = {};
  firestoreMap.benchmarkInfo = makeCompatable(result);
  firestoreMap.date = getDate();

  return firestoreMap;
}

/**
 * This function makes the result object returned from benchmark app aligned
 * with target firestore collection's schema.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function makeCompatableWithFirestore(result) {
  addGpuInfo(result);
  serializeTensors(result);
  return result;
}

/**
 * Append GPU info to device name.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function addGpuInfo(result) {
  const gpuInfo = result.gpuInfo;
  delete result.gpuInfo;
  if (gpuInfo == null || gpuInfo === 'MISS') {
    return;
  }

  if (result.deviceInfo.device == null) {
    result.deviceInfo.device = `(GPU: ${gpuInfo})`;
  } else {
    result.deviceInfo.device = `${result.deviceInfo.device} (GPU: ${gpuInfo})`;
  }
  return result;
}

/**
 * Benchmark results contain tensors that are represented as nested arrays.
 * Nested arrays are not supported on Firestore, so they are serialized
 * before they are stored.
 *
 * @param result Individual result in a list of fulfilled promises
 */
function serializeTensors(result) {
  let kernels = result.memoryInfo.kernels;
  for (kernel of kernels) {
    kernel.inputShapes = JSON.stringify(kernel.inputShapes);
    kernel.outputShapes = JSON.stringify(kernel.outputShapes);
  }
  return result;
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
exports.makeCompatableWithFirestore = makeCompatableWithFirestore;
exports.addGpuInfo = addGpuInfo;
exports.serializeTensors = serializeTensors;
exports.getReadableDate = getReadableDate;
exports.formatForFirestore = formatForFirestore;
exports.runFirestore = runFirestore;
exports.endFirebaseInstance = endFirebaseInstance;
