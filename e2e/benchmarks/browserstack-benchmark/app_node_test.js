const fs = require('fs');
const {benchmark, write, getOneBenchmarkResult, runBenchmarkFromFile} =
    require('./app.js');
const {
  addResultToFirestore,
  serializeTensors,
  getReadableDate,
  formatForFirestore,
  runFirestore,
  firebaseConfig
} = require('./firestore.js');

describe('test app.js cli', () => {
  const filePath = './benchmark_test_results.json';
  let config;
  let mockRunOneBenchmark;
  let failMockRunOneBenchmark;
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

    // Bypasses BrowserStack with preset successful mock results
    mockRunOneBenchmark =
        jasmine.createSpy('mockRunOneBenchmark').and.callFake((tabId) => {
          return Promise.resolve(mockResults[tabId]);
        });

    // Bypasses Browserstack with preset failed mock results
    failMockRunOneBenchmark =
        jasmine.createSpy('mockRunOneBenchmark').and.callFake((tabId) => {
          return Promise.reject(`Error: ${tabId} failed.`);
        });

    // Before each spec, create a mock benchmark and set testing browser
    // configuration this helps ensure that everything is set to the expected
    // contents before the spec is run
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

  it('benchmark function benchmarks each browser-device pairing ', async () => {
    // Receives list of promises from benchmark function call
    const testResults = await benchmark(config, mockRunOneBenchmark);

    // Extracts value results from promises, effectively formatting
    const formattedResults = {};
    for (let i = 0; i < Object.keys(config.browsers).length; i++) {
      await new Promise(resolve => {
        const result = testResults[i].value;
        formattedResults[result.tabId] = result;
        return resolve();
      });
    }

    // Expected mockRunOneBenchmark stats
    expect(mockRunOneBenchmark.calls.count())
        .toEqual(Object.keys(config.browsers).length);
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('iPhone_XS_1', undefined);
    expect(mockRunOneBenchmark)
        .toHaveBeenCalledWith('Samsung_Galaxy_S20_1', undefined);
    expect(mockRunOneBenchmark).toHaveBeenCalledWith('Windows_10_1', undefined);
    expect(mockRunOneBenchmark)
        .toHaveBeenCalledWith('OS_X_Catalina_1', undefined);

    // Expected value from promise all
    expect(formattedResults).toEqual(mockResults);
  });

  it('getOneBenchmark rejects if a benchmark consistently fails', async () => {
    // Expected failed mock benchmark results
    await expectAsync(
        getOneBenchmarkResult('iPhone_XS_1', 3, failMockRunOneBenchmark))
        .toBeRejectedWith(`Error: iPhone_XS_1 failed.`);

    // Expected mock function call stats
    expect(failMockRunOneBenchmark.calls.count()).toEqual(3);
  });

  it('getOneBenchmark fulfills if a benchmark fails and then succeeds',
     async () => {
       /* Bypasses Browserstack with preset results. Benchmark will fail on the
        * call, but succeed on the second call */
       let called = false;
       const failThenSucceedMockRunOneBenchmark =
           jasmine.createSpy('mockRunOneBenchmark').and.callFake((tabId) => {
             if (called) {
               return mockRunOneBenchmark(tabId);
             }
             called = true;
             return failMockRunOneBenchmark(tabId);
           });

       // Gets a successful benchmark result
       const succeedBenchmarkResult = await getOneBenchmarkResult(
           'iPhone_XS_1', 3, failThenSucceedMockRunOneBenchmark);

       // Expected mock function call stats
       expect(failMockRunOneBenchmark.calls.count()).toEqual(1);
       expect(mockRunOneBenchmark.calls.count()).toEqual(1);

       // Expected successful mock benchmark results
       expect(succeedBenchmarkResult).toEqual(mockResults.iPhone_XS_1);
     });

  it('getOneBenchmark fulfills if a benchmark succeeds immediately',
     async () => {
       // Gets a successful benchmark result
       const succeedBenchmarkResult =
           await getOneBenchmarkResult('iPhone_XS_1', 3, mockRunOneBenchmark);

       // Expected mock funciton call stats
       expect(mockRunOneBenchmark.calls.count()).toEqual(1);

       // Expected successful mock benchmark results
       expect(succeedBenchmarkResult).toEqual(mockResults.iPhone_XS_1);
     });

  it('checks that the benchmark is being run with the correct JSON', () => {
    runBenchmarkFromFile(testingConfig, mockBenchmark);
    expect(mockBenchmark).toHaveBeenCalledWith(testingConfig);
  });
});

describe('test adding to firestore', () => {
  let db;
  let mockDb;
  let mockResultValue;

  beforeAll(async () => {
    // Set a longer jasmine timeout than 5 seconds
    jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;

    // References result collection and checks credentials
    db = await runFirestore(firebaseConfig);
  });

  beforeEach(() => {
    // mockResultValue is the result of a successful benchmark
    mockResultValue = require('./firestore_test_value.json');
    mockDb = spyOn(db, 'add');
    mockSerialization = jasmine.createSpy('mockSerialization');
    mockDate = jasmine.createSpy('mockDate').and.returnValue('7/21/2021');
  });

  it('Expects db.add to be called with formatted results', () => {
    mockDb.and.returnValue(Promise.resolve({id: 123}));
    let expectedAdd = {
      result:
          formatForFirestore(mockResultValue, serializeTensors, getReadableDate)
    };
    addResultToFirestore(db, mockResultValue.tabId, mockResultValue);
    expect(db.add).toHaveBeenCalledWith(expectedAdd);
  });

  it('Expects a date key to exist and have the correct value', () => {
    let testFormat =
        formatForFirestore(mockResultValue, mockSerialization, mockDate);
    expect(testFormat.date).toEqual('7/21/2021');
  });

  it('Expects serialization to cover all nested arrays', () => {
    const mockSerializedResults =
        formatForFirestore(mockResultValue, serializeTensors, mockDate);
    for (kernel of mockSerializedResults.benchmarkInfo.memoryInfo.kernels) {
      expect(typeof (kernel.inputShapes)).toEqual('string');
      expect(typeof (kernel.outputShapes)).toEqual('string');
    }
  });
});
