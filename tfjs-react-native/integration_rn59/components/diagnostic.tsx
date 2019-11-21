/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import React, { Fragment } from 'react';
import { Button, SafeAreaView, StyleSheet, ScrollView, View, StatusBar } from 'react-native';

import * as tf from '@tensorflow/tfjs';
import { Run } from './run';
import { simpleOpRunner, precisionTestRunner, mobilenetRunner, localModelRunner, trainModelRunner, saveModelRunner, localGraphModelRunner } from './ml';

interface ScreenProps {
  returnToMain: () => void;
}

export class Diagnostic extends React.Component<ScreenProps> {
  constructor(props: ScreenProps) {
    super(props);
  }

  render() {
    return (
      <Fragment>
        <StatusBar barStyle='dark-content' />
        <SafeAreaView>
          <ScrollView
            contentInsetAdjustmentBehavior='automatic'
            style={styles.scrollView}>

            <View style={styles.body}>
              <View style={styles.sectionContainer}>
                <Button
                  onPress={this.props.returnToMain}
                  title='Back'
                />
              </View>

              <View style={styles.sectionContainer}>
                <Run label='tf.getBackend()' result={`${tf.getBackend()}`}>
                </Run>
                <Run label='tf.version_core' result={`${tf.version_core}`}>
                </Run>
                <Run label='WEBGL_VERSION' result={`${
                  tf.env().getNumber('WEBGL_VERSION')}`}>
                </Run>
                <Run label='WEBGL_RENDER_FLOAT32_ENABLED' result={`${
                  tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED')}`}>
                </Run>
                <Run label='WEBGL_DOWNLOAD_FLOAT_ENABLED' result={`${
                  tf.env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')}`}>
                </Run>
                <Run label='WEBGL_BUFFER_SUPPORTED' result={`${
                  tf.env().getNumber('WEBGL_BUFFER_SUPPORTED')}`}>
                </Run>
                <Run label='new Float32Array([2,3, NaN])' result={`${
                  new Float32Array([2, 3, NaN])
                  }`}>
                </Run>
                <Run label='SimpleOp tf.square(3)'
                  getRunner={simpleOpRunner} numRuns={1}></Run>
                <Run label='tf.scalar(2.4).square()'
                  getRunner={precisionTestRunner} numRuns={1}></Run>
                <Run label='mobilenet'
                  getRunner={mobilenetRunner} numRuns={1}></Run>
                <Run label='bundleStorageIO'
                  getRunner={localModelRunner} numRuns={1}></Run>
                <Run label='bundleStorageIO - graph model'
                  getRunner={localGraphModelRunner} numRuns={1}></Run>
                <Run label='train model'
                  getRunner={trainModelRunner} numRuns={1}></Run>
                <Run label='asyncStorareIO'
                  getRunner={saveModelRunner} numRuns={1}></Run>
              </View>
            </View>
          </ScrollView>
        </SafeAreaView>
      </Fragment>
    );
  }
}

const styles = StyleSheet.create({
  scrollView: {
    backgroundColor: 'white',
  },
  body: {
    backgroundColor: 'white',
    marginBottom: 20,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: 'black',
    marginBottom: 6,
  },
});
