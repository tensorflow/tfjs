const {execFile} = require('child_process');
const { config } = require('process');
const app = require('./app.js');
const runBenchmarkFromFile = app.runBenchmarkFromFile;

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

describe("Testing the benchmarks flag", () => {
  let mockBenchmark;

  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

    //create a mock benchmark function because only the argument is being tested
    mockBenchmark = jasmine.createSpy('mockBenchmark');
    testingConfig = require('./test_config.json');
    runBenchmarkFromFile(testingConfig, mockBenchmark);
  });

  it("checks that the benchmark function is called", () => {
    expect(mockBenchmark).toHaveBeenCalled();
  });

  it("checks that the JSON is loaded correctly", () => {
    expect(mockBenchmark).toHaveBeenCalledWith(testingConfig);
  });
});
