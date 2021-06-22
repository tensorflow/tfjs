const {execFile} = require('child_process');
const fs = require('fs');
const {benchmark, write} = require('./app.js');

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
    expect(await exec('echo', ['-n', 'Hello, world!'])).toMatch('Hello');
  });

  it('runs a command that takes a while', async () => {
    expect(await exec('sleep', ['0.1'])).toEqual('');
  });

  it('runs a command that fails', async () => {
    await expectAsync(exec('exit 1')).toBeRejected();
  });
});

const mockResults = {
  'iPhone_XS_1': {
    timeInfo: {
      times: [216.00000000000045],
      averageTime: 216.00000000000045,
      minTime: 216.00000000000045,
      maxTime: 216.00000000000045
    },
    tabId: 'iPhone_XS_1'
  },
  'Samsung_Galaxy_S20_1': {
    timeInfo: {
      times: [428.89999999897555],
      averageTime: 428.89999999897555,
      minTime: 428.89999999897555,
      maxTime: 428.89999999897555
    },
    tabId: 'Samsung_Galaxy_S20_1'
  },
  'Windows_10_1': {
    timeInfo: {
      times: [395.8500000001095],
      averageTime: 395.8500000001095,
      minTime: 395.8500000001095,
      maxTime: 395.8500000001095
    },
    tabId: 'Windows_10_1'
  },
  'OS_X_Catalina_1': {
    timeInfo: {
      times: [176.19500000728294],
      averageTime: 176.19500000728294,
      minTime: 176.19500000728294,
      maxTime: 176.19500000728294
    },
    tabId: 'OS_X_Catalina_1'
  }
};

describe('tests outfile writing capabilities', () => {
  const filePath = './benchmark_test_results.json';

  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

    // Writes to mock results file
    write(filePath, mockResults);
  });

  it('waits for file to be written to, then checks for accuracy', () => {
    setTimeout(() => fs.readFile(filePath, 'utf8', (err, data) => {
      expect(data).toEqual(mockResults);
    }), 1000);
  });
});

describe('tests benchmark capabilities', () => {
  const config = {
    benchmark: {model: 'mobilenet_v2', numRuns: 1, backend: 'wasm'},
    browsers: {
      iPhone_XS_1: {
        base: 'BrowserStack',
        browser: 'iphone',
        browser_version: 'null',
        os: 'ios',
        os_version: '12',
        device: 'iPhone XS',
        real_mobile: true
      },
      Samsung_Galaxy_S20_1: {
        base: 'BrowserStack',
        browser: 'android',
        browser_version: 'null',
        os: 'android',
        os_version: '10.0',
        device: 'Samsung Galaxy S20',
        real_mobile: true
      },
      Windows_10_1: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: '84.0',
        os: 'Windows',
        os_version: '10',
        device: null
      },
      OS_X_Catalina_1: {
        base: 'BrowserStack',
        browser: 'chrome',
        browser_version: '84.0',
        os: 'OS X',
        os_version: 'Catalina',
        device: null
      }
    }
  };
  let mockRunOneBenchmark;
  let formattedResults;

  beforeAll(async () => {
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

    // Bypasses BrowserStack with preset mock results
    mockRunOneBenchmark =
        jasmine.createSpy('mockRunOneBenchmark').and.callFake((tabId) => {
          return new Promise((resolve) => {resolve(mockResults[tabId])});
        });

    // Receives and formats results from benchmark function call
    let testResults = await benchmark(config, mockRunOneBenchmark);
    formattedResults = {};
    for (let i = 0; i < Object.keys(config.browsers).length; i++) {
      await new Promise(resolve => {
        const result = testResults[i]['value'];
        formattedResults[result['tabId']] = result;
        return resolve();
      });
    }
  });

  it('checks run stats of mocked function', () => {
    expect(mockRunOneBenchmark.calls.count())
        .toEqual(Object.keys(config.browsers).length);
    expect(mockRunOneBenchmark)
        .toHaveBeenCalledWith(
            'iPhone_XS_1' || 'Samsung_Galaxy_S20_1' || 'Windows_10_1' ||
            'OS_X_Catalina_1');
  });

  it('checks output', () => {
    expect(formattedResults).toEqual(mockResults);
  });
});
