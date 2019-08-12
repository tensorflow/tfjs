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

import {describeWithFlags} from '../../jasmine_util';
import {WEBGL_ENVS} from './backend_webgl_test_registry';
import {dotify, getLogicalCoordinatesFromFlatIndex} from './shader_compiler_util';

describeWithFlags('shader compiler', WEBGL_ENVS, () => {
  it('dotify takes two arrays of coordinates and produces' +
         'the glsl that finds the dot product of those coordinates',
     () => {
       const coords1 = ['r', 'g', 'b', 'a'];
       const coords2 = ['x', 'y', 'z', 'w'];

       expect(dotify(coords1, coords2))
           .toEqual('dot(vec4(r,g,b,a), vec4(x,y,z,w))');
     });

  it('dotify should split up arrays into increments of vec4s', () => {
    const coords1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
    const coords2 = ['h', 'i', 'j', 'k', 'l', 'm', 'n'];

    expect(dotify(coords1, coords2))
        .toEqual(
            'dot(vec4(a,b,c,d), vec4(h,i,j,k))+dot(vec3(e,f,g), vec3(l,m,n))');
  });

  it('getLogicalCoordinatesFromFlatIndex produces glsl that takes' +
         'a flat index and finds its coordinates within that shape',
     () => {
       const coords = ['r', 'c', 'd'];
       const shape = [1, 2, 3];

       expect(getLogicalCoordinatesFromFlatIndex(coords, shape))
           .toEqual(
               'int r = index / 6; index -= r * 6;' +
               'int c = index / 3; int d = index - c * 3;');
     });
});
