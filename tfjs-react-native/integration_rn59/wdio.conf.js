const MILLIS_PER_MIN = 60 * 1000;

const sharedCapabilities = Object.freeze({
  'browserstack.debug': true,
  'browserstack.local': true,
  'browserstack.networkLogs': true,
});

exports.config = {
  before: function() {
    require('ts-node').register({files: true})
  },

  //
  // ====================
  // Runner Configuration
  // ====================
  runner: 'local',
  // =====================
  // Server Configurations
  // =====================
  hostname: 'hub-cloud.browserstack.com',
  port: 443,
  // path: '/wd/hub',
  services: ['browserstack'],
  user: process.env.BROWSERSTACK_USERNAME,
  key: process.env.BROWSERSTACK_KEY,
  // bridge network requests to the machine that started the test session.
  browserstackLocal: true,
  specs: ['./test/**/*.ts'],
  exclude: [],
  maxInstances: 5,
  capabilities: [
    Object.assign(
        {
          'app': 'deeplearnjs1/MyApp',
          'device': 'Samsung Galaxy S9 Plus',
          // 'device': 'Google Pixel 3',
          'os': 'android',
          'os_version': '9.0',
          'build': 'tfjs-react-native android',
          'name': 'React Native Integration Test (Android)',
        },
        sharedCapabilities),
  ],
  // Level of logging verbosity: trace | debug | info | warn | error | silent
  logLevel: 'warn',
  // If you only want to run your tests until a specific amount of tests have
  // failed use bail (default is 0 - don't bail, run all tests).
  bail: 0,
  // Default timeout for all waitFor* commands.
  waitforTimeout: 10000,
  // Default timeout in milliseconds for request
  connectionRetryTimeout: 180000,
  // Default request retries count
  connectionRetryCount: 3,
  framework: 'jasmine',
  reporters: ['spec'],
  jasmineNodeOpts: {
    defaultTimeoutInterval: MILLIS_PER_MIN * 6,
  },
}
