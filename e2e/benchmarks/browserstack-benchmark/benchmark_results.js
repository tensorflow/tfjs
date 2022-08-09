let res = [
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [12, 3, 4, 4, 6, 3],
        'averageTime': 5.333333333333333,
        'minTime': 3,
        'maxTime': 12
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 4.9999999999990905,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 4.9999999999990905}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'OS_X_Monterey_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'safari',
        'browser_version': '15.3',
        'os': 'OS X',
        'os_version': 'Monterey',
        'device': null
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          13.199999999254942, 10.5, 10.400000000372529, 7.900000000372529,
          8.199999999254942, 6.900000000372529
        ],
        'averageTime': 9.516666666604578,
        'minTime': 6.900000000372529,
        'maxTime': 13.199999999254942
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 0.189083,
          'extraInfo': 'LinearReadProgram: 0.189083'
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 0.189083}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'OS_X_Monterey_2',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'OS X',
        'os_version': 'Monterey',
        'browser': 'chrome',
        'device': null,
        'browser_version': '103.0'
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [14, 12, 11.999999999999773, 12, 12, 12],
        'averageTime': 12.333333333333295,
        'minTime': 11.999999999999773,
        'maxTime': 14
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 12,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 12}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_13_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'ios',
        'os_version': '15',
        'browser': 'iphone',
        'device': 'iPhone 13 Pro Max',
        'browser_version': null,
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          22, 20.000000000000455, 16.999999999999545, 15.000000000000455, 16,
          13.999999999999545
        ],
        'averageTime': 17.333333333333332,
        'minTime': 13.999999999999545,
        'maxTime': 22
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 14,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 14}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_13_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'ios',
        'os_version': '15',
        'browser': 'iphone',
        'device': 'iPhone 13',
        'browser_version': null,
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [275, 279, 261, 261.00000000000045, 259.99999999999955, 266],
        'averageTime': 267,
        'minTime': 259.99999999999955,
        'maxTime': 279
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 265.99999999999955,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 265.99999999999955}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_12_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 12 Pro Max',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [262, 258, 259.0000000000002, 267.9999999999998, 266, 267],
        'averageTime': 263.3333333333333,
        'minTime': 258,
        'maxTime': 267.9999999999998
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 260,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 260}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_12_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 12',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          24, 341.0000000000002, 312.9999999999998, 314.0000000000002,
          314.9999999999998, 307.00000000000045
        ],
        'averageTime': 269.00000000000006,
        'minTime': 24,
        'maxTime': 341.0000000000002
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 307,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 307}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_11_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 11 Pro Max',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [325, 321, 328, 314, 329, 321.00000000000045],
        'averageTime': 323.00000000000006,
        'minTime': 314,
        'maxTime': 329
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 318,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 318}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          42.00000000000023, 35.99999999999977, 35, 34.00000000000023,
          36.99999999999977, 37
        ],
        'averageTime': 36.833333333333336,
        'minTime': 34.00000000000023,
        'maxTime': 42.00000000000023
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 39.000000000000455,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 39.000000000000455}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_XS_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '15',
        'device': 'iPhone XS',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          71.20000000018626, 71.10000000009313, 65.79999999981374,
          67.60000000009313, 66.39999999990687, 61.299999999813735
        ],
        'averageTime': 67.23333333331782,
        'minTime': 61.299999999813735,
        'maxTime': 71.20000000018626
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 60.39999999990687,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 60.39999999990687}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_6_Pro_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Google Pixel 6 Pro',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          937.8000000000466, 936, 936.6999999999534, 938.7000000001863,
          937.0999999998603, 942.0999999998603
        ],
        'averageTime': 938.0666666666511,
        'minTime': 936,
        'maxTime': 942.0999999998603
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 941.5999999998603,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 941.5999999998603}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_5_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Google Pixel 5',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          13.400000000372529, 13.700000000186265, 15, 14.899999999441206,
          13.899999999441206, 13.600000000558794
        ],
        'averageTime': 14.083333333333334,
        'minTime': 13.400000000372529,
        'maxTime': 15
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 13.800000000745058,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 13.800000000745058}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_S22_Ultra_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Samsung Galaxy S22 Ultra',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          531.3999999761581, 538.7999999523163, 537.8999999761581, 539, 532.5,
          538.8999999761581
        ],
        'averageTime': 536.4166666467985,
        'minTime': 531.3999999761581,
        'maxTime': 539
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 536.8999999761581,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 536.8999999761581}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_M52_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Samsung Galaxy M52',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          1077.4000000357628, 1089.1000000238419, 1079.6000000238419,
          1113.300000011921, 1103, 1105.300000011921
        ],
        'averageTime': 1094.6166666845481,
        'minTime': 1077.4000000357628,
        'maxTime': 1113.300000011921
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 1104.2000000476837,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 1104.2000000476837}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_A52_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Samsung Galaxy A52',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          2800.3000000715256, 2174.399999976158, 21.199999928474426,
          1.7000000476837158, 1.4000000953674316, 1.100000023841858
        ],
        'averageTime': 833.3500000238419,
        'minTime': 1.100000023841858,
        'maxTime': 2800.3000000715256
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 2.5,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 2.5}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Xiaomi_Redmi_Note_11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Xiaomi Redmi Note 11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          110.30000001192093, 82.09999999403954, 79.09999999403954,
          77.8999999910593, 76.29999999701977, 76.5
        ],
        'averageTime': 83.69999999801318,
        'minTime': 76.29999999701977,
        'maxTime': 110.30000001192093
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 76.6000000089407,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 76.6000000089407}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_Note_20_Ultra_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Samsung Galaxy Note 20 Ultra',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          31.800000000745058, 8.900000000372529, 5.199999999254942,
          3.099999999627471, 3.200000001117587, 3.200000001117587
        ],
        'averageTime': 9.233333333705863,
        'minTime': 3.099999999627471,
        'maxTime': 31.800000000745058
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 5.100000001490116,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 5.100000001490116}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_A11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Samsung Galaxy A11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          413.59999999403954, 416.20000000298023, 419.69999998807907,
          416.70000000298023, 417.69999998807907, 415.70000000298023
        ],
        'averageTime': 416.5999999965231,
        'minTime': 413.59999999403954,
        'maxTime': 419.69999998807907
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 416.09999999403954,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 416.09999999403954}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_4_XL_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Google Pixel 4 XL',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          299.5, 299.1000000005588, 295.8999999994412, 295.20000000018626,
          293.69999999925494, 294.3999999994412
        ],
        'averageTime': 296.29999999981374,
        'minTime': 293.69999999925494,
        'maxTime': 299.5
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 295.29999999981374,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 295.29999999981374}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 1, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_S10_Plus_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '9.0',
        'device': 'Samsung Galaxy S10 Plus',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },

  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [8, 8, 4.0000000000009095, 2.9999999999990905, 3, 6],
        'averageTime': 5.333333333333333,
        'minTime': 2.9999999999990905,
        'maxTime': 8
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 3,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 3}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'OS_X_Monterey_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'safari',
        'browser_version': '15.3',
        'os': 'OS X',
        'os_version': 'Monterey',
        'device': null
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          14.299999997019768, 10, 10.600000001490116, 7.699999995529652,
          6.800000004470348, 6.800000004470348
        ],
        'averageTime': 9.366666667163372,
        'minTime': 6.800000004470348,
        'maxTime': 14.299999997019768
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 0.189666,
          'extraInfo': 'LinearReadProgram: 0.189666'
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 0.189666}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'OS_X_Monterey_2',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'OS X',
        'os_version': 'Monterey',
        'browser': 'chrome',
        'device': null,
        'browser_version': '103.0'
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [12.999999999999886, 13, 12, 11, 13, 11],
        'averageTime': 12.166666666666648,
        'minTime': 11,
        'maxTime': 13
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 12,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 12}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_13_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'ios',
        'os_version': '15',
        'browser': 'iphone',
        'device': 'iPhone 13 Pro Max',
        'browser_version': null,
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [21, 14.000000000000455, 14.999999999999545, 15, 15, 14],
        'averageTime': 15.666666666666666,
        'minTime': 14,
        'maxTime': 21
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 14,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 14}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_13_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'os': 'ios',
        'os_version': '15',
        'browser': 'iphone',
        'device': 'iPhone 13',
        'browser_version': null,
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [263, 261, 270, 268, 272, 260],
        'averageTime': 265.6666666666667,
        'minTime': 260,
        'maxTime': 272
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 258,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 258}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_12_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 12 Pro Max',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [273.00000000000045, 270.99999999999955, 260, 256, 261, 272],
        'averageTime': 265.5,
        'minTime': 256,
        'maxTime': 273.00000000000045
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 273,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 273}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_12_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 12',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [326, 322.00000000000045, 318.99999999999955, 315, 318, 320],
        'averageTime': 320,
        'minTime': 315,
        'maxTime': 326
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 321.0000000000009,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 321.0000000000009}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_11_Pro_Max_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 11 Pro Max',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [319, 320, 322, 323, 315, 320],
        'averageTime': 319.8333333333333,
        'minTime': 315,
        'maxTime': 323
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 317,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 317}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '14',
        'device': 'iPhone 11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [41, 34, 35, 35, 40, 40.00000000000023],
        'averageTime': 37.500000000000036,
        'minTime': 34,
        'maxTime': 41
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 39,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 39}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'iPhone_XS_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'iphone',
        'browser_version': null,
        'os': 'ios',
        'os_version': '15',
        'device': 'iPhone XS',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          69.29999999981374, 60.90000000037253, 64.8999999994412, 62.5, 62,
          69.40000000037253
        ],
        'averageTime': 64.83333333333333,
        'minTime': 60.90000000037253,
        'maxTime': 69.40000000037253
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 66.1000000005588,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 66.1000000005588}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_6_Pro_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Google Pixel 6 Pro',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          940.1000000000931, 940.1000000000931, 945.1999999997206,
          949.7000000001863, 946.5, 951.2000000001863
        ],
        'averageTime': 945.4666666667132,
        'minTime': 940.1000000000931,
        'maxTime': 951.2000000001863
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 946,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 946}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_5_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Google Pixel 5',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [10, 10.5, 11, 11.5, 12.299999952316284, 11.100000023841858],
        'averageTime': 11.066666662693024,
        'minTime': 10,
        'maxTime': 12.299999952316284
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 10.600000023841858,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 10.600000023841858}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_S22_Ultra_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '12.0',
        'device': 'Samsung Galaxy S22 Ultra',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          529.6999999880791, 531, 532.5, 531.9000000059605, 539.9000000059605,
          539.5999999940395
        ],
        'averageTime': 534.0999999990066,
        'minTime': 529.6999999880791,
        'maxTime': 539.9000000059605
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 533.5,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 533.5}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_M52_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Samsung Galaxy M52',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          1123.0999999642372, 1137.199999988079, 1133.2999999523163,
          1150.199999988079, 1134.1000000238419, 1157.2000000476837
        ],
        'averageTime': 1139.1833333273728,
        'minTime': 1123.0999999642372,
        'maxTime': 1157.2000000476837
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 1146.800000011921,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 1146.800000011921}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_A52_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Samsung Galaxy A52',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          2843, 2843.2000000476837, 2835.100000023842, 2838.6999999284744, 2847,
          2146
        ],
        'averageTime': 2725.5,
        'minTime': 2146,
        'maxTime': 2847
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 57,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 57}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Xiaomi_Redmi_Note_11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '11.0',
        'device': 'Xiaomi Redmi Note 11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          100.19999998807907, 87.29999999701977, 87.90000000596046,
          88.70000000298023, 87.29999999701977, 86.5
        ],
        'averageTime': 89.64999999850988,
        'minTime': 86.5,
        'maxTime': 100.19999998807907
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 86.90000000596046,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 86.90000000596046}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_Note_20_Ultra_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Samsung Galaxy Note 20 Ultra',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          43.19999999925494, 9.699999999254942, 5.5, 3.099999999627471,
          2.600000001490116, 2.900000000372529
        ],
        'averageTime': 11.166666666666666,
        'minTime': 2.600000001490116,
        'maxTime': 43.19999999925494
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 5.300000000745058,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 5.300000000745058}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_A11_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Samsung Galaxy A11',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          414.8999999910593, 428.30000001192093, 422.5, 427.20000000298023,
          425.20000000298023, 421.79999999701977
        ],
        'averageTime': 423.31666666766006,
        'minTime': 414.8999999910593,
        'maxTime': 428.30000001192093
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 427,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 427}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Google_Pixel_4_XL_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '10.0',
        'device': 'Google Pixel 4 XL',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    'status': 'fulfilled',
    'value': {
      'timeInfo': {
        'times': [
          301, 294.69999998807907, 296, 294.30000001192093, 303.5,
          296.89999997615814
        ],
        'averageTime': 297.7333333293597,
        'minTime': 294.30000001192093,
        'maxTime': 303.5
      },
      'memoryInfo': {
        'newBytes': 0,
        'newTensors': 0,
        'peakBytes': 8388612,
        'kernels': [{
          'name': 'Conv2D',
          'bytesAdded': 4194304,
          'totalBytesSnapshot': 8388612,
          'tensorsAdded': 1,
          'totalTensorsSnapshot': 3,
          'inputShapes': [[1, 1, 1024, 1024], [1, 1, 1, 1]],
          'outputShapes': [[1, 1, 1024, 1024]],
          'kernelTimeMs': 294.80000001192093,
          'extraInfo': ''
        }],
        'kernelNames': ['Conv2D'],
        'aggregatedKernels': [{'name': 'Conv2D', 'timeMs': 294.80000001192093}]
      },
      'codeSnippet':
          '() => {\n    return tf.conv2d(image, filter, 2, \'same\', \'NCHW\');\n  }',
      'tabId': 'Samsung_Galaxy_S10_Plus_1',
      'deviceInfo': {
        'base': 'BrowserStack',
        'browser': 'android',
        'browser_version': null,
        'os': 'android',
        'os_version': '9.0',
        'device': 'Samsung Galaxy S10 Plus',
        'real_mobile': true
      },
      'modelInfo': {'model': 'codeSnippet', 'numRuns': 6, 'backend': 'webgl'}
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          8,
          6,
          4,
          3.0000000000009095,
          5.9999999999990905,
          3
        ],
        "averageTime": 5,
        "minTime": 3,
        "maxTime": 8
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 3,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 3
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "OS_X_Monterey_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "safari",
        "browser_version": "15.3",
        "os": "OS X",
        "os_version": "Monterey",
        "device": null
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          14.600000023841858,
          11.800000011920929,
          12.399999976158142,
          9.400000035762787,
          11.100000023841858,
          8.399999976158142
        ],
        "averageTime": 11.283333341280619,
        "minTime": 8.399999976158142,
        "maxTime": 14.600000023841858
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 1.681916,
            "extraInfo": "SquareReadProgram: 1.681916"
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 1.681916
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "OS_X_Monterey_2",
      "deviceInfo": {
        "base": "BrowserStack",
        "os": "OS X",
        "os_version": "Monterey",
        "browser": "chrome",
        "device": null,
        "browser_version": "103.0"
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          14,
          12,
          12,
          11.000000000000227,
          12.999999999999773,
          12
        ],
        "averageTime": 12.333333333333334,
        "minTime": 11.000000000000227,
        "maxTime": 14
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 12,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 12
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_13_Pro_Max_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "os": "ios",
        "os_version": "15",
        "browser": "iphone",
        "device": "iPhone 13 Pro Max",
        "browser_version": null,
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          17,
          14,
          14,
          14.000000000000227,
          14.999999999999773,
          14
        ],
        "averageTime": 14.666666666666666,
        "minTime": 14,
        "maxTime": 17
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 14,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 14
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_13_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "os": "ios",
        "os_version": "15",
        "browser": "iphone",
        "device": "iPhone 13",
        "browser_version": null,
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          262,
          277,
          276,
          269,
          261,
          261
        ],
        "averageTime": 267.6666666666667,
        "minTime": 261,
        "maxTime": 277
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 259,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 259
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_12_Pro_Max_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "iphone",
        "browser_version": null,
        "os": "ios",
        "os_version": "14",
        "device": "iPhone 12 Pro Max",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          278.00000000000045,
          258,
          256.99999999999955,
          257,
          277,
          270
        ],
        "averageTime": 266.1666666666667,
        "minTime": 256.99999999999955,
        "maxTime": 278.00000000000045
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 259.9999999999991,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 259.9999999999991
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_12_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "iphone",
        "browser_version": null,
        "os": "ios",
        "os_version": "14",
        "device": "iPhone 12",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          321,
          320,
          308,
          307,
          319,
          309.00000000000045
        ],
        "averageTime": 314.00000000000006,
        "minTime": 307,
        "maxTime": 321
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 310,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 310
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_11_Pro_Max_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "iphone",
        "browser_version": null,
        "os": "ios",
        "os_version": "14",
        "device": "iPhone 11 Pro Max",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          327,
          318,
          314.0000000000002,
          324.9999999999998,
          328,
          318
        ],
        "averageTime": 321.6666666666667,
        "minTime": 314.0000000000002,
        "maxTime": 328
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 325,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 325
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_11_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "iphone",
        "browser_version": null,
        "os": "ios",
        "os_version": "14",
        "device": "iPhone 11",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          41.99999999999977,
          34,
          33,
          34,
          38,
          38
        ],
        "averageTime": 36.499999999999964,
        "minTime": 33,
        "maxTime": 41.99999999999977
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 38,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 38
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "iPhone_XS_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "iphone",
        "browser_version": null,
        "os": "ios",
        "os_version": "15",
        "device": "iPhone XS",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          65,
          68,
          61.799999999813735,
          59.799999999813735,
          57.30000000074506,
          58.5
        ],
        "averageTime": 61.73333333339542,
        "minTime": 57.30000000074506,
        "maxTime": 68
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 60,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 60
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Google_Pixel_6_Pro_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "12.0",
        "device": "Google Pixel 6 Pro",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          938.3000000000466,
          946.5,
          944.4000000001397,
          946.8000000000466,
          948.5,
          950.1999999999534
        ],
        "averageTime": 945.7833333333643,
        "minTime": 938.3000000000466,
        "maxTime": 950.1999999999534
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 942.3000000000466,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 942.3000000000466
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Google_Pixel_5_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "12.0",
        "device": "Google Pixel 5",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          15,
          12.400000005960464,
          11.800000004470348,
          11,
          10.5,
          16.5
        ],
        "averageTime": 12.866666668405136,
        "minTime": 10.5,
        "maxTime": 16.5
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 10.899999998509884,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 10.899999998509884
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_S22_Ultra_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "12.0",
        "device": "Samsung Galaxy S22 Ultra",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          530.5,
          528.1000000238419,
          547,
          530,
          549.9000000357628,
          532.3000000119209
        ],
        "averageTime": 536.3000000119209,
        "minTime": 528.1000000238419,
        "maxTime": 549.9000000357628
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 536.6000000238419,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 536.6000000238419
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_M52_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "11.0",
        "device": "Samsung Galaxy M52",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          1038,
          1055.3999999761581,
          1051.4000000357628,
          1056.1000000238419,
          1067,
          1074
        ],
        "averageTime": 1056.9833333392937,
        "minTime": 1038,
        "maxTime": 1074
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 1076,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 1076
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_A52_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "11.0",
        "device": "Samsung Galaxy A52",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          2806.3000000715256,
          2796.7999999523163,
          2805.899999976158,
          2043,
          64.10000002384186,
          9.399999976158142
        ],
        "averageTime": 1754.25,
        "minTime": 9.399999976158142,
        "maxTime": 2806.3000000715256
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 3.6999999284744263,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 3.6999999284744263
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Xiaomi_Redmi_Note_11_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "11.0",
        "device": "Xiaomi Redmi Note 11",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          122.5,
          86.90000000596046,
          88.6000000089407,
          88.1000000089407,
          86,
          84.8999999910593
        ],
        "averageTime": 92.83333333581686,
        "minTime": 84.8999999910593,
        "maxTime": 122.5
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 88,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 88
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_Note_20_Ultra_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "10.0",
        "device": "Samsung Galaxy Note 20 Ultra",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          24.200000000186265,
          6.2999999998137355,
          4.2000000001862645,
          3,
          2.400000000372529,
          3.199999999254942
        ],
        "averageTime": 7.216666666635622,
        "minTime": 2.400000000372529,
        "maxTime": 24.200000000186265
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 4.099999999627471,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 4.099999999627471
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_A11_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "10.0",
        "device": "Samsung Galaxy A11",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          418.19999999925494,
          414.30000000074506,
          413.90000000037253,
          415.30000000074506,
          419.19999999925494,
          418.09999999962747
        ],
        "averageTime": 416.5,
        "minTime": 413.90000000037253,
        "maxTime": 419.19999999925494
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 416.5,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 416.5
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Google_Pixel_4_XL_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "10.0",
        "device": "Google Pixel 4 XL",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  },
  {
    "status": "fulfilled",
    "value": {
      "timeInfo": {
        "times": [
          292.59999999962747,
          279.7999999988824,
          282.90000000037253,
          283.2000000011176,
          282.80000000074506,
          281
        ],
        "averageTime": 283.71666666679084,
        "minTime": 279.7999999988824,
        "maxTime": 292.59999999962747
      },
      "memoryInfo": {
        "newBytes": 0,
        "newTensors": 0,
        "peakBytes": 8388612,
        "kernels": [
          {
            "name": "Conv2D",
            "bytesAdded": 4194304,
            "totalBytesSnapshot": 8388612,
            "tensorsAdded": 1,
            "totalTensorsSnapshot": 3,
            "inputShapes": [
              [
                1,
                1,
                1024,
                1024
              ],
              [
                1,
                1,
                1,
                1
              ]
            ],
            "outputShapes": [
              [
                1,
                1,
                1024,
                1024
              ]
            ],
            "kernelTimeMs": 285.69999999925494,
            "extraInfo": ""
          }
        ],
        "kernelNames": [
          "Conv2D"
        ],
        "aggregatedKernels": [
          {
            "name": "Conv2D",
            "timeMs": 285.69999999925494
          }
        ]
      },
      "codeSnippet": "() => {\n    return tf.conv2d(image, filter, 3, 'same', 'NCHW');\n  }",
      "tabId": "Samsung_Galaxy_S10_Plus_1",
      "deviceInfo": {
        "base": "BrowserStack",
        "browser": "android",
        "browser_version": null,
        "os": "android",
        "os_version": "9.0",
        "device": "Samsung Galaxy S10 Plus",
        "real_mobile": true
      },
      "modelInfo": {
        "model": "codeSnippet",
        "numRuns": 6,
        "backend": "webgl"
      }
    }
  }

];
