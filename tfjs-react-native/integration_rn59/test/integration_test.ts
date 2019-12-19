const MILLIS_PER_MIN = 60 * 1000;
const INITIAL_LOAD_TIMEOUT = 5 * MILLIS_PER_MIN;
const TEST_RUN_TIMEOUT = 5 * MILLIS_PER_MIN;
const TEST_POLL_INTERVAL = 2000;
const DEFAULT_TIMEOUT = 30000;

describe('tfjs-core unit tests', () => {
  it('unit tests should pass', async () => {
    const driver = browser;

    const unitTestBtn = await driver.$('~unit-test-btn');
    await unitTestBtn.waitForExist(
        INITIAL_LOAD_TIMEOUT, false,
        'Could not find unit test button. \n' +
            'The browserstacklocal tunnel was likely not established');
    await unitTestBtn.waitForEnabled(INITIAL_LOAD_TIMEOUT);
    await unitTestBtn.click();

    const backendNameEl = await driver.$('~backendName');
    await backendNameEl.waitForExist(DEFAULT_TIMEOUT);
    const backendName = await backendNameEl.getText();
    expect(backendName).toEqual('backend=rn-webgl');

    // Wait for the unit tests to complete.
    const testCompleteEl = await driver.$('~testComplete');
    await driver.waitUntil(async () => {
      const testStatus = await testCompleteEl.getText();
      return testStatus === 'testsComplete=true';
    }, TEST_RUN_TIMEOUT, 'Unit tests timed out', TEST_POLL_INTERVAL);

    // Get the number of passed and run tests
    const passedTestsEl = await driver.$('~passedTests');
    await passedTestsEl.waitForExist(DEFAULT_TIMEOUT);
    const passedTestsText = await passedTestsEl.getText();

    const parts = passedTestsText.match(/(?<passed>\d+) of (?<total>\d+)/);
    expect(parts).not.toBeNull();

    if (parts != null && parts.groups != null) {
      const passed = parseInt(parts.groups.passed, 10);
      const total = parseInt(parts.groups.total, 10);
      const failed = total - passed;

      if (failed > 0) {
        fail(`${failed} tests failed out of ${total}`);
        const failureMessages = await driver.$('~failureMessages');
        await failureMessages.waitForExist(DEFAULT_TIMEOUT);
        const failureMessagesText = await failureMessages.getText();

        const TEST_SEPARATOR = '###';
        const TITLE_SEPARATOR = '%%%';

        const testFailures = failureMessagesText.split(TEST_SEPARATOR);
        for (let i = 0; i < testFailures.length; i++) {
          const testFailure = testFailures[i];
          const [title, messages] = testFailure.split(TITLE_SEPARATOR);

          fail(`${title}\n${messages}`);
        }
      }
    }
  }, TEST_RUN_TIMEOUT + INITIAL_LOAD_TIMEOUT);
});

// Mark as a module for ts compiler.
export {};
