const benchmarkResults = [
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "ANGLE (Google, Vulkan 1.2.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)", "tabId": "Windows_11_1", "deviceInfo": { "base": "BrowserStack", "browser": "chrome", "browser_version": "103.0", "os": "Windows", "os_version": "11", "device": null }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "ANGLE (Google, Vulkan 1.2.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)", "tabId": "Windows_11_2", "deviceInfo": { "base": "BrowserStack", "browser": "edge", "browser_version": "103.0", "os": "Windows", "os_version": "11", "device": null }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "firefox", "browser_version": "103.0", "os": "Windows", "os_version": "11", "device": null }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "firefox", "browser_version": "103.0", "os": "Windows", "os_version": "7", "device": null }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Apple GPU", "tabId": "OS_X_Monterey_1", "deviceInfo": { "base": "BrowserStack", "browser": "safari", "browser_version": "15.3", "os": "OS X", "os_version": "Monterey", "device": null }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "ANGLE (Intel Inc., Intel(R) UHD Graphics 630, OpenGL 4.1)", "tabId": "OS_X_Monterey_2", "deviceInfo": { "base": "BrowserStack", "os": "OS X", "os_version": "Monterey", "browser": "chrome", "device": null, "browser_version": "103.0" }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Apple GPU", "tabId": "iPhone_13_Pro_Max_1", "deviceInfo": { "base": "BrowserStack", "os": "ios", "os_version": "15", "browser": "iphone", "device": "iPhone 13 Pro Max", "browser_version": null, "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Apple GPU", "tabId": "iPhone_13_1", "deviceInfo": { "base": "BrowserStack", "os": "ios", "os_version": "15", "browser": "iphone", "device": "iPhone 13", "browser_version": null, "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "iphone", "browser_version": null, "os": "ios", "os_version": "14", "device": "iPhone 12 Pro Max", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "iphone", "browser_version": null, "os": "ios", "os_version": "14", "device": "iPhone 12", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "iphone", "browser_version": null, "os": "ios", "os_version": "14", "device": "iPhone 11 Pro Max", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "error": "Error: WebGL2 is unavailable!", "deviceInfo": { "base": "BrowserStack", "browser": "iphone", "browser_version": null, "os": "ios", "os_version": "14", "device": "iPhone 11", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Apple GPU", "tabId": "iPhone_XS_1", "deviceInfo": { "base": "BrowserStack", "browser": "iphone", "browser_version": null, "os": "ios", "os_version": "15", "device": "iPhone XS", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Mali-G78", "tabId": "Google_Pixel_6_Pro_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "12.0", "device": "Google Pixel 6 Pro", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 620", "tabId": "Google_Pixel_5_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "12.0", "device": "Google Pixel 5", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "ANGLE (Samsung Xclipse 920) on Vulkan 1.1.179", "tabId": "Samsung_Galaxy_S22_Ultra_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "12.0", "device": "Samsung Galaxy S22 Ultra", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 642L", "tabId": "Samsung_Galaxy_M52_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "11.0", "device": "Samsung Galaxy M52", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 618", "tabId": "Samsung_Galaxy_A52_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "11.0", "device": "Samsung Galaxy A52", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 610", "tabId": "Xiaomi_Redmi_Note_11_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "11.0", "device": "Xiaomi Redmi Note 11", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Mali-G77", "tabId": "Samsung_Galaxy_Note_20_Ultra_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "10.0", "device": "Samsung Galaxy Note 20 Ultra", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 506", "tabId": "Samsung_Galaxy_A11_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "10.0", "device": "Samsung Galaxy A11", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 8, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Adreno (TM) 640", "tabId": "Google_Pixel_4_XL_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "10.0", "device": "Google Pixel 4 XL", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
  ,
  { "status": "fulfilled", "value": { "mrtInfo": 4, "timeInfo": null, "memoryInfo": null, "gpuInfo": "Mali-G76", "tabId": "Samsung_Galaxy_S10_Plus_1", "deviceInfo": { "base": "BrowserStack", "browser": "android", "browser_version": null, "os": "android", "os_version": "9.0", "device": "Samsung Galaxy S10 Plus", "real_mobile": true }, "modelInfo": { "model": "getMaxDrawBuffers", "numRuns": 1, "backend": "webgl" } } }
]
