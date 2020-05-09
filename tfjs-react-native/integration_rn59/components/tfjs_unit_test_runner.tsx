
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

// @ts-ignore
import jasmineRequire from 'jasmine-core/lib/jasmine-core/jasmine.js';
// tslint:disable-next-line: no-imports-from-dist
import * as jasmine_util from '@tensorflow/tfjs-core/dist/jasmine_util';
import React, { Component, Fragment } from 'react';
import { StyleSheet, Text, View, ViewStyle, ScrollView } from 'react-native';
import * as tf from '@tensorflow/tfjs-core';

interface TestRunnerProps {
  backend: string;
}

interface FailedTestInfo {
  suiteName?: string;
  testName: string;
  failedExpectations: Array<{ message: string, stack: string }>;
  message?: string;
}

interface TestRunnerState {
  passedTests: number;
  failedTests: FailedTestInfo[];
  totalTests: number;
  testsComplete: boolean;
  testsStarted: boolean;
  backendName?: string;
}

export class TestRunner extends Component<TestRunnerProps, TestRunnerState> {
  constructor(props: TestRunnerProps) {
    super(props);
    this.state = {
      passedTests: 0,
      failedTests: [],
      totalTests: 0,
      testsComplete: false,
      testsStarted: false,
    };
  }

  /**
   * If a getRunner function is present in props, execute it
   * numRun times and report the result and timing info.
   */
  async componentDidMount() {
    const backendName = tf.getBackend();
    this.setState({
      backendName,
    });

    let passedTests = 0;
    const failedTests: FailedTestInfo[] = [];

    // A lot of the code below is adapted from
    // node_modules/jasmine-core/lib/jasmine-core/boot.js
    // it provides a custom way to start jasmine in the RN app.

    // Helper function for adding jasmine functionlaity to global.
    // tslint:disable-next-line: no-any
    function extend(destination: any, source: any) {
      for (const property in source) {
        destination[property] = source[property];
      }
      return destination;
    }

    const jasmine = jasmineRequire.core(jasmineRequire);
    // @ts-ignore
    global.jasmine = jasmine;
    const env: jasmine.Env = jasmine.getEnv();

    const jasmineInterface = jasmineRequire.interface(jasmine, env);
    extend(global, jasmineInterface);

    //@ts-ignore
    env.configure({
      random: false,
      specFilter: (spec: jasmine.Spec) => {
        const name = spec.getFullName();

        if(name.match('Backend registration')) {
          // These tests use spyOn on a module. Skip them until they
          // can be updated. Coverage for this functionality is tested
          // directly in this package.
          return false;
        }
        if(name.match('isMobile')) {
          // Browser specific mobile test
          return false;
        }
        return true;
      }
     });

    // Custom reporter to collect the test results
    const reactReporter: jasmine.CustomReporter = {
      jasmineStarted: suiteInfo => {
        // The console.warn below seems necessary in order for the spy on
        // console.warn defined in one of the tests to run corrently.
        console.warn('starting tests');
        //@ts-ignore
        console.reportErrorsAsExceptions = false;
        this.setState({
          testsStarted: true,
          totalTests: suiteInfo.totalSpecsDefined,
        });
      },
      specDone: result => {
        if (result.failedExpectations == null ||
          result.failedExpectations.length === 0) {
          passedTests += 1;
          this.setState({
            passedTests,
          });
        }
        else if (result.failedExpectations.length > 0) {
          const failureInfo: FailedTestInfo = {
            testName: result.fullName,
            failedExpectations: result.failedExpectations.map(f => {

              return {
                message: f.message,
                stack: f.stack,
              };
            }),
          };
          // Log to console to make it easier to view these in dev tools.
          console.log('Test Failure');
          console.log(JSON.stringify(failureInfo, null, 2));
          failedTests.push(failureInfo);
          this.setState({
            failedTests,
          });
        }
      },
      jasmineDone: () => {
        //@ts-ignore
        console.reportErrorsAsExceptions = true;
        this.setState({
          testsComplete: true,
          failedTests,
        });
      }
    };
    env.addReporter(reactReporter);

    jasmine_util.setTestEnvs(
      [{
        name: 'test-rn',
        backendName: this.props.backend,
        flags: {
          'WEBGL_CPU_FORWARD': false, 'WEBGL_SIZE_UPLOAD_UNIFORM': 0,
        }
      }]);

    // import tests
    require('@tensorflow/tfjs-core/dist/tests');

    // import tfjs-react-native-tests;
    require('@tensorflow/tfjs-react-native/dist/tests');

    // Start the test runner
    env.execute();
  }

  renderTestFailure(failure: FailedTestInfo, key: number) {
    return <View style={styles.failedTest} key={key}>
      <Text style={styles.failedTestName}>
        {failure.testName}
      </Text>
      <Fragment>
        {failure.failedExpectations.map((expecation, i) => <Text key={i}>
          {expecation.message}
        </Text>)}
      </Fragment>
    </View>;
  }

  formatTestFailuresForWebDriver(failures: FailedTestInfo[]) {
    const TEST_SEPARATOR = '###';
    const TITLE_SEPARATOR = '%%%';
    return failures.map((failure, i) => {
      const testName = failure.testName;
      const messages = failure.failedExpectations.map((expecation, i) =>
        expecation.message).join('\n');

      return `${testName}${TITLE_SEPARATOR}${messages}`;
    }).join(TEST_SEPARATOR);
  }

  render() {
    const { passedTests, failedTests, totalTests, backendName, testsComplete }
      = this.state;

    return (
      <Fragment>
        <View testID='testInfo'
          accessibilityLabel='testInfo'
          style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}>Info</Text>
          <Text testID='backendName' accessibilityLabel='backendName'>
            backend={backendName}
          </Text>
          <Text testID='testComplete' accessibilityLabel='testComplete'>
            testsComplete={String(testsComplete)}
          </Text>
          <Text>tf.env().platformName={tf.env().platformName}</Text>
        </View>
        <View style={styles.sectionContainer}>
          <Text style={styles.sectionTitle}
            testID='passedTests'
            accessibilityLabel='passedTests'>
            Passed Tests {passedTests} of {totalTests}
          </Text>
        </View>
        <View>
          <Text style={styles.failureMessages}
            testID='failureMessages'
            accessibilityLabel='failureMessages'>
            {this.formatTestFailuresForWebDriver(failedTests)}
          </Text>
        </View>
        <ScrollView contentInsetAdjustmentBehavior='automatic'>
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionTitle}>
              Failed Tests ({failedTests.length})
            </Text>
            <View
              testID='failedTests'
              accessibilityLabel='failedTests'>
              {failedTests.map((f, i) => this.renderTestFailure(f, i))}
            </View>
          </View>
        </ScrollView>
      </Fragment>
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
};

const styles = StyleSheet.create({
  container,
  containerMounted,
  failedTest: {
    backgroundColor: 'white',
    margin: 5,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  failedTestName: {
    backgroundColor: 'white',
    fontWeight: 'bold',
  },
  failureMessages: {
    fontSize: 1,
  },
  sectionContainer: {
    marginTop: 24,
    paddingHorizontal: 12,
    paddingBottom: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: 'black',
    marginBottom: 6,
  },
});
