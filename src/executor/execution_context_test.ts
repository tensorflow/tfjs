/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {ExecutionContext} from './execution_context';

let context: ExecutionContext;
describe('ExecutionContext', () => {
  beforeEach(() => {
    context = new ExecutionContext({});
  });
  afterEach(() => {});

  it('should initialize', () => {
    expect(context.currentContext).toEqual([
      {id: 0, frameName: '', iterationId: 0}
    ]);
    expect(context.currentContextId).toEqual('');
  });

  describe('enterFrame', () => {
    it('should add new Frame', () => {
      context.enterFrame('1');
      expect(context.currentContextId).toEqual('/1-0');
      expect(context.currentContext).toEqual([
        {id: 0, frameName: '', iterationId: 0},
        {id: 1, frameName: '1', iterationId: 0}
      ]);
    });
  });

  describe('exitFrame', () => {
    it('should remove Frame', () => {
      context.enterFrame('1');
      context.exitFrame();

      expect(context.currentContextId).toEqual('');
      expect(context.currentContext).toEqual([
        {id: 0, frameName: '', iterationId: 0}
      ]);
    });

    it('should remember previous Frame', () => {
      context.enterFrame('1');
      context.nextIteration();
      context.enterFrame('2');
      context.exitFrame();

      expect(context.currentContextId).toEqual('/1-1');
      expect(context.currentContext).toEqual([
        {id: 0, frameName: '', iterationId: 0},
        {id: 2, frameName: '1', iterationId: 1}
      ]);
    });
  });

  describe('nextIteration', () => {
    it('should increate iteration', () => {
      context.enterFrame('1');
      context.nextIteration();

      expect(context.currentContextId).toEqual('/1-1');
      expect(context.currentContext).toEqual([
        {id: 0, frameName: '', iterationId: 0},
        {id: 2, frameName: '1', iterationId: 1}
      ]);
    });
  });
});
