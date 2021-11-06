/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {enableProdMode} from '@angular/core';
import {platformBrowserDynamic} from '@angular/platform-browser-dynamic';

import {AppModule, DevAppModule} from './app/app/app.module';
import {environment} from './environments/environment';

// Load AppModule in production mode.
if (environment.production) {
  enableProdMode();
  platformBrowserDynamic().bootstrapModule(AppModule).catch(
      err => console.error(err));
}
// Load DevAppModule in dev mode.
//
// DevAppModule has extra modules (e.g. StoreDevtoolsModule) registered for
// debugging purpose.
else {
  platformBrowserDynamic()
      .bootstrapModule(DevAppModule)
      .catch(err => console.error(err));
}
