
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

// import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';
// @ts-ignore
import jasmineRequire from 'jasmine-core/lib/jasmine-core/jasmine.js';
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';
import React, { Component } from 'react';
import { StyleSheet, Text, View, ViewStyle } from 'react-native';

interface TestRunnerProps {
}

interface FailedTestInfo {
  suiteName?: string;
  testName: string;
  failedExpectations: string[]
  message?: string;
}

interface TestRunnerState {
  passedTests: number;
  failedTests: FailedTestInfo[];
  testsComplete: boolean;
  testsStarted: boolean;
}

export class TestRunner extends Component<TestRunnerProps, TestRunnerState> {
  constructor(props: TestRunnerProps) {
    super(props);
    this.state = {
      passedTests: 0,
      failedTests: [],
      testsComplete: false,
      testsStarted: false,
    }
  }

  /**
   * If a getRunner function is present in props, execute it
   * numRun times and report the result and timing info.
   */
  async componentDidMount() {
    let passedTests = 0;
    const failedTests: FailedTestInfo[] = [];

    // Helper function for adding jasmine functionlaity to global.
    function extend(destination: any, source: any) {
      for (var property in source) destination[property] = source[property];
      return destination;
    }

    const jasmine = jasmineRequire.core(jasmineRequire);
    // @ts-ignore
    global.jasmine = jasmine;
    const env: jasmine.Env = jasmine.getEnv();

    const jasmineInterface = jasmineRequire.interface(jasmine, env);
    extend(global, jasmineInterface);

    // Custom reporter to collect the test results
    const reactReporter: jasmine.CustomReporter = {
      jasmineStarted: suiteInfo => {
        this.setState({
          testsStarted: true,
        });
      },
      // suiteStarted: result => {},
      // specStarted: result => { },
      specDone: result => {
        if (result.failedExpectations == null || result.failedExpectations.length === 0) {
          passedTests += 1;
          this.setState({
            passedTests,
          });
        }
        else if (result.failedExpectations.length > 0) {
          const failureInfo: FailedTestInfo = {
            testName: result.fullName,
            failedExpectations: result.failedExpectations.map(f => f.message),
          };
          failedTests.push(failureInfo);
          this.setState({
            failedTests,
          });
        }
      },
      // suiteDone: (result) => {},
      jasmineDone: () => {
        this.setState({
          testsComplete: true,
        });
      }
    };
    env.addReporter(reactReporter);

    // TODO. Fix
    jasmine_util.setTestEnvs(
      [{ name: 'test-rn', backendName: 'cpu', flags: {} }]);

    // import tests
    require('@tensorflow/tfjs-core/dist/tests');

    // Start the test runner
    env.execute();
  }

  render() {
    const { passedTests, failedTests } = this.state;

    return (
      <View style={styles.labelArea}>
        <Text style={styles.labelHeader}>Label</Text>
        <Text style={styles.labelText}>{passedTests}</Text>
        <Text style={styles.labelText}>{failedTests.length}</Text>
      </View>
    );
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
}
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
