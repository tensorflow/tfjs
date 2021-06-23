const {execFile} = require('child_process');
const {config} = require('process');
const {runBenchmarkFromFile} = require('./app.js');

function exec(command, args) {
  return new Promise((fulfill, reject) => {
    execFile(command, args, (error, stdout) => {
      if (error) {
        reject(error);
      } else {
        fulfill(stdout);
      }
    });
  });
}

describe('benchmark app cli', () => {
  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  it('runs a command line command', async () => {
    expect(await exec('echo', ['-n', 'Hello, world!']))
        .toMatch('Hello');
  });

  it('runs a command that takes a while', async () => {
    expect(await exec('sleep', ['0.1'])).toEqual('');
  });

  it('runs a command that fails', async () => {
    await expectAsync(exec('exit 1')).toBeRejected();
  });
});

describe("Testing the --benchmarks argument", () => {
  let mockBenchmark;

  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  /*
  before each spec, create a mock benchmark and set testing browser configuration
  this helps ensure that everything is set to the expected contents before the spec is run
  */
  beforeEach(() => {
    mockBenchmark = jasmine.createSpy('mockBenchmark');
    testingConfig = require('./test_config.json');
  });

  it("checks that the benchmark function is called", () => {
    runBenchmarkFromFile(testingConfig, mockBenchmark);
    expect(mockBenchmark).toHaveBeenCalled();
  });

  it("checks that the benchmark is being run with the correct JSON", () => {
    runBenchmarkFromFile(testingConfig, mockBenchmark);
    expect(mockBenchmark).toHaveBeenCalledWith(testingConfig);
  });
});
