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

import { h } from 'preact';
import { render } from 'preact-render-spy';

import { VisorComponent } from './visor';
import { SurfaceInfoStrict } from '../types';

afterEach(() => {
  document.body.innerHTML = '';
});

describe('Visor Component', () => {
  it('renders an empty visor', () => {
    const wrapper = render(
      <VisorComponent surfaceList={[]} />
    );

    expect(wrapper.find('.visor').length).toBe(1);
    expect(wrapper.find('.visor-surfaces').length).toBe(1);
    expect(wrapper.find('.tf-surface').length).toBe(0);
    expect(wrapper.state().isOpen).toBe(true);
    expect(wrapper.state().isFullscreen).toBe(false);
  });

  it('renders an empty and closed visor', () => {
    const wrapper = render(
      <VisorComponent
        surfaceList={[]}
        startOpen={false}
      />
    );

    expect(wrapper.find('.visor').length).toBe(1);
    expect(wrapper.state().isOpen).toBe(false);
    expect(wrapper.state().isFullscreen).toBe(false);
  });

  it('renders a surface', () => {
    const surfaceList: SurfaceInfoStrict[] = [
      { name: 'surface 1', tab: 'tab 1' },
    ];

    const wrapper = render(
      <VisorComponent surfaceList={surfaceList} />
    );

    expect(wrapper.find('.tf-surface').length).toBe(1);
    expect(wrapper.find('.tf-surface').text()).toMatch('surface 1');
    expect(wrapper.find('.tf-tab').length).toBe(1);
    expect(wrapper.find('.tf-tab').text()).toMatch('tab 1');
  });

  it('switches tabs on click', () => {
    const surfaceList: SurfaceInfoStrict[] = [
      { name: 'surface 1', tab: 'tab 1' },
      { name: 'surface 2', tab: 'tab 2' },
    ];

    const wrapper = render(
      <VisorComponent surfaceList={surfaceList} />
    );

    expect(wrapper.find('.tf-tab').length).toBe(2);
    expect(wrapper.state().activeTab).toEqual('tab 2');

    // Clicks
    wrapper.find('.tf-tab').at(0).simulate('click');
    expect(wrapper.state().activeTab).toEqual('tab 1');
    expect(wrapper.find('.tf-tab').at(0).attr('data-isactive' as never))
      .toEqual(true);
    expect(wrapper.find('.tf-tab').at(1).attr('data-isactive' as never))
      .toEqual(false);

    expect(wrapper.find('.tf-surface').at(0).attr('data-visible' as never))
      .toEqual(true);
    expect(wrapper.find('.tf-surface').at(1).attr('data-visible' as never))
      .toEqual(false);

    wrapper.find('.tf-tab').at(1).simulate('click');
    expect(wrapper.state().activeTab).toEqual('tab 2');
    expect(wrapper.find('.tf-tab').at(0).attr('data-isactive' as never))
      .toEqual(false);
    expect(wrapper.find('.tf-tab').at(1).attr('data-isactive' as never))
      .toEqual(true);

    expect(wrapper.find('.tf-surface').at(0).attr('data-visible' as never))
      .toEqual(false);
    expect(wrapper.find('.tf-surface').at(1).attr('data-visible' as never))
      .toEqual(true);
  });

  it('hides on close button click', () => {
    const surfaceList: SurfaceInfoStrict[] = [];

    const wrapper = render(
      <VisorComponent surfaceList={surfaceList} />
    );

    expect(wrapper.state().isOpen).toEqual(true);

    const hideButton = wrapper.find('.visor-controls').children().at(1);
    expect(hideButton.text()).toEqual('Hide');

    hideButton.simulate('click');
    expect(wrapper.state().isOpen).toEqual(false);
  });

  it('maximises and minimizes', () => {
    const surfaceList: SurfaceInfoStrict[] = [];

    const wrapper = render(
      <VisorComponent surfaceList={surfaceList} />
    );

    expect(wrapper.state().isOpen).toEqual(true);

    let toggleButton;
    toggleButton = wrapper.find('.visor-controls').children().at(0);
    expect(toggleButton.text()).toEqual('Maximise');
    expect(wrapper.state().isFullscreen).toEqual(false);
    expect(wrapper.find('.visor').at(0).attr('data-isfullscreen' as never))
      .toEqual(false);

    toggleButton.simulate('click');
    toggleButton = wrapper.find('.visor-controls').children().at(0);
    expect(toggleButton.text()).toEqual('Minimize');
    expect(wrapper.state().isFullscreen).toEqual(true);
    expect(wrapper.find('.visor').at(0).attr('data-isfullscreen' as never))
      .toEqual(true);
  });
});
