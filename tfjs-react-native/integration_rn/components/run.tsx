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

import React, { Component } from 'react';
import { StyleSheet, Text, View, ViewStyle } from 'react-native';

interface RunProps {
  getRunner?: () => Promise<() => Promise<string>>;
  numRuns?: number;
  result?: string;
  label: string;
}

interface RunState {
  mountComplete: boolean;
  computedResult?: string;
  lastPredictionTime?: number;
  avgPredictionTime?: number;
}

export class Run extends Component<RunProps, RunState> {
  constructor(props: RunProps) {
    super(props);
    this.state = {
      mountComplete: false,
    };
  }

  /**
   * If a getRunner function is present in props, execute it
   * numRun times and report the result and timing info.
   */
  async componentDidMount() {
    const { getRunner, numRuns } = this.props;
    let computedResult;
    let time;
    const numberOfRuns = numRuns != null ? numRuns : 1;
    if (getRunner != null) {
      const runner = await getRunner();
      const start = new Date();
      for (let index = 0; index < numberOfRuns; index++) {
        computedResult = await runner();
      }
      const end = new Date();
      time = end.getMilliseconds() - start.getMilliseconds();
    }
    this.setState({
      mountComplete: true,
      computedResult,
      lastPredictionTime: time ? time : undefined,
      avgPredictionTime: time ? (time / numberOfRuns) : undefined,
    });
  }

  render() {
    const {
      mountComplete,
      computedResult,
      lastPredictionTime,
      avgPredictionTime,
    } = this.state;

    const { label, result, numRuns } = this.props;
    const res = computedResult ? computedResult : result;
    return (
      <View style={mountComplete ? styles.containerMounted : styles.container}>
        <View style={styles.labelArea}>
          <Text style={styles.labelHeader}>Label</Text>
          <Text style={styles.labelText}>{label}</Text>
        </View>
        <View style={styles.resultArea}>
          <View style={styles.row}>
            {lastPredictionTime != null ?
              <Text style={styles.labelHeader}>
                Total Time: {lastPredictionTime}
              </Text>
              : undefined}
            {numRuns != null ?
              <Text style={styles.labelHeader}>
                Avg: {avgPredictionTime} ({numRuns})
              </Text>
              : undefined}
          </View>
          <Text style={styles.resultText}>{JSON.stringify(res)}</Text>
        </View>
      </View>);
  }
}

const container: ViewStyle = {
  display: 'flex',
  flexDirection: 'row',
  backgroundColor: '#FFFDE7',
  padding: 5,
  marginBottom: 5,
};

const containerMounted: ViewStyle = {
  ...container,
  backgroundColor: '#C8E6C9',
};
const styles = StyleSheet.create({
  container,
  containerMounted,
  row: {
    display: 'flex',
    flexDirection: 'row',
  },
  labelArea: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    minWidth: 100,
  },
  labelHeader: {
    fontSize: 10,
    marginRight: 4,
  },
  labelText: {
    fontSize: 12
  },
  resultArea: {
    marginLeft: 5,
    paddingLeft: 5,
    borderLeftColor: '#444',
    borderLeftWidth: 1,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
  },
  resultText: {
    fontSize: 14
  }
});
