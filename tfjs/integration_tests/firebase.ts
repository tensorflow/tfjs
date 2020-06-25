/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {BenchmarkLog} from './types';

// tslint:disable-next-line:no-any
declare let __karma__: any;

import 'firebase/auth';
import 'firebase/database';
// tslint:disable-next-line:max-line-length
import {ApplicationConfig, BenchmarkRunEntry, BenchmarkEntry, BenchmarkHashes} from './firebase_types';

export const karmaFlags = parseKarmaFlags(__karma__.config.args);

const config: ApplicationConfig = {
  apiKey: karmaFlags.apiKey,
  authDomain: 'jstensorflow.firebaseapp.com',
  databaseURL: 'https://tensorflowjs-benchmarks.firebaseio.com',
  projectId: 'jstensorflow',
  storageBucket: 'jstensorflow.appspot.com',
  messagingSenderId: '433613381222'
};

firebase.initializeApp(config);
try {
  firebase.auth();
} catch (e) {
  throw new Error(`Firebase auth failed with error: ${e}`);
}

export async function logBenchmarkRun(
    benchmarkName: string, logs: BenchmarkLog[]): Promise<void> {
  const date = new Date();
  let month = (date.getMonth() + 1).toString();
  if (month.length === 1) {
    month = '0' + month;
  }
  let day = date.getDate().toString();
  if (day.length === 1) {
    day = '0' + day;
  }
  const humanReadableDate = `${date.getFullYear()}-${month}-${day}`;

  const runs: {[params: string]: BenchmarkRunEntry} = {};
  logs.forEach(log => {
    runs[log.params] = {averageTimeMs: log.averageTimeMs};
  });
  const entry: BenchmarkEntry = {
    userAgent: navigator.userAgent,
    runs,
    timestamp: Date.now(),
    hashes: karmaFlags.hashes
  };

  if (navigator.hardwareConcurrency != null) {
    entry.hardwareConcurrency = navigator.hardwareConcurrency;
  }

  const entryDisplay: string = JSON.stringify(entry, undefined, 2);
  const ref = `${humanReadableDate}/${benchmarkName}/${karmaFlags.browsers}`;
  if (!karmaFlags.nightly) {
    console.log('Not nightly so not querying firebase. Would have added: ');
    console.log(ref);
    console.log(entryDisplay);
  } else {
    console.log('Writing to firebase:');
    console.log(ref);
    console.log(entryDisplay);
    return new Promise<void>(resolve => {
      firebase.database()
          .ref(ref)
          // We set the database entry to be an array of one value so in the
          // future we can benchmark multiple devices.
          .set(entry, error => {
            if (error) {
              throw new Error(`Write to firebase failed with error:
              ${error}`);
            }
            resolve();
          });
    });
  }
}

interface KarmaFlags {
  apiKey: string;
  nightly: boolean;
  browsers: string;
  hashes: BenchmarkHashes;
}

export function parseKarmaFlags(args: string[]): KarmaFlags {
  let apiKey: string;
  let nightly = false;
  let browsers: string;
  let hashes: {};
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--firebaseKey') {
      apiKey = args[i + 1];
    }
    if (args[i] === '--nightly') {
      nightly = true;
    }
    if (args[i] === '--browsers') {
      browsers = args[i + 1];
    }
    if (args[i] === '--hashes') {
      hashes = JSON.parse(args[i + 1]);
    }
  }
  return {apiKey, nightly, browsers, hashes: hashes || {}};
}
