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

import { h, Component } from 'preact';
import { css } from 'glamor';
import { SurfaceInfoStrict, StyleOptions } from '../types';

// Internal Props
interface SurfaceProps extends SurfaceInfoStrict {
  visible: boolean;
  registerSurface: (name: string, tab: string, surface: SurfaceComponent)
    => void;
}

/**
 * A surface is container for visualizations and other rendered thigns.
 * It consists of a containing DOM Element, a label and an empty drawArea.
 */
export class SurfaceComponent extends Component<SurfaceProps> {

  static defaultStyles: Partial<StyleOptions> = {
    maxWidth: '550px',
    maxHeight: '580px',
    height: 'auto',
    width: 'auto',
  };

  container: HTMLElement;
  label: HTMLElement;
  drawArea: HTMLElement;

  componentDidMount() {
    const { name, tab } = this.props;
    this.props.registerSurface(name, tab, this);
  }

  componentDidUpdate() {
    // Prevent re-rendering of this component as it
    // is primarily controlled outside of this class
    return false;
  }

  render() {
    const { name, visible, styles } = this.props;
    const finalStyles = {
      ...SurfaceComponent.defaultStyles,
      ...styles,
    };

    const { width, height, } = finalStyles;
    let { maxHeight, maxWidth, } = finalStyles;
    maxHeight = height === SurfaceComponent.defaultStyles.height ?
      maxHeight : height;
    maxWidth = width === SurfaceComponent.defaultStyles.width ?
      maxWidth : width;

    const surfaceStyle = css({
      display: visible ? 'block' : 'none',
      backgroundColor: 'white',
      marginTop: '10px',
      marginBottom: '10px',
      boxShadow: '0 0 6px -3px #777',
      padding: '10px !important',
      height,
      width,
      maxHeight,
      maxWidth,
      overflow: 'auto',
    });

    const labelStyle = css({
      backgroundColor: 'white',
      boxSizing: 'border-box',
      borderBottom: '1px solid #357EDD',
      lineHeight: '2em',
      marginBottom: '20px',
      fontWeight: '600',
      textAlign: 'center',
    });

    const drawAreaStyle = css({
      boxSizing: 'border-box',
    });

    return (
      <div
        className={`${surfaceStyle} tf-surface`}
        ref={(r) => this.container = r}
        data-visible={visible}
      >
        <div className={`${labelStyle} tf-label`} ref={(r) => this.label = r}>
          {name}
        </div>

        <div
          className={`${drawAreaStyle} tf-draw-area`}
          ref={(r) => this.drawArea = r}
        />
      </div>
    );
  }
}
