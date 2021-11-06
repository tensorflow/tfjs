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

import {NgModule} from '@angular/core';
import {BrowserModule} from '@angular/platform-browser';
import {RouterModule} from '@angular/router';
import {routerReducer, StoreRouterConnectingModule} from '@ngrx/router-store';
import {StoreModule} from '@ngrx/store';
import {StoreDevtoolsModule} from '@ngrx/store-devtools';

import {appReducers} from '../store/state';

import {AppComponent} from './app.component';

/** The main application module. */
@NgModule({
  declarations: [AppComponent],
  imports: [
    BrowserModule,
    StoreModule.forRoot({
      router: routerReducer,
      ...appReducers,
    }),
    RouterModule.forRoot([
      {path: '', component: AppComponent},
      {path: '**', redirectTo: ''},
    ]),
    StoreRouterConnectingModule.forRoot(),
  ],
  bootstrap: [AppComponent]
})
export class AppModule {
}

/**
 * The DevAppModule adds the NgRx dev tools support when running in dev
 * mode.
 *
 * Download the chrome extension that works with internal sites at:
 * go/redux-devtools
 */
@NgModule({
  imports: [
    AppModule,
    StoreDevtoolsModule.instrument({
      maxAge: 200,      // Retains last 200 states
      autoPause: true,  // Pauses recording actions and state changes when the
                        // extension window is not open
    }),
  ],
  bootstrap: [AppComponent]
})
export class DevAppModule {
}
