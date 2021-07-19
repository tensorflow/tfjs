const fs = require('fs');
const {benchmark, write, runBenchmarkFromFile} = require('./app.js');

describe('test app.js cli', () => {
  const filePath = './benchmark_test_results.json';
  let config;
  let mockRunOneBenchmark;
  let mockResults;
  let mockBenchmark;

  beforeAll(() => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  });

  beforeEach(() => {
    // Preset mock results and corresponding config
    mockResults = {
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
    config = {
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

    // Bypasses BrowserStack with preset mock results
    mockRunOneBenchmark =
        jasmine.createSpy('mockRunOneBenchmark').and.callFake((tabId) => {
          return Promise.resolve(mockResults[tabId]);
        });

    /*
    before each spec, create a mock benchmark and set testing browser configuration
    this helps ensure that everything is set to the expected contents before the spec is run
    */
    mockBenchmark = jasmine.createSpy('mockBenchmark');
    testingConfig = require('./test_config.json');
  })

  it('checks for outfile accuracy', async () => {
    // Writes to mock results file
    await write(filePath, mockResults);

    fs.readFile(filePath, 'utf8', (err, data) => {
      expect(data).toEqual(JSON.stringify(mockResults, null, 2));
    });
  });

  it('checks mocked function and consequent value of promise all', async () => {
    // Receives list of promises from benchmark function call
    const testResults = await benchmark(config, mockRunOneBenchmark);

    // Extracts value results from promises, effectively formatting
    const formattedResults = {};
    for (let i = 0; i < Object.keys(config.browsers).length; i++) {
      await new Promise(resolve => {
        const result = testResults[i]['value'];
        formattedResults[result['tabId']] = result;
        return resolve();
      });
    }

    // Expected mockRunOneBenchmark stats
    expect(mockRunOneBenchmark.calls.count())
        .toEqual(Object.keys(config.browsers).length);
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('iPhone_XS_1');
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('Samsung_Galaxy_S20_1');
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('Windows_10_1');
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('OS_X_Catalina_1');

    // Expected value from promise all
    expect(formattedResults).toEqual(mockResults);
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
