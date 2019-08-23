/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import React, { Fragment } from 'react';
import { StyleSheet, View, Image, Text, TouchableHighlight, TouchableOpacity } from 'react-native';
import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import {StyleTranfer} from './style_transfer';
import {imageUrlToTensor, tensorToImageUrl, resizeImage} from './image_utils';
import * as tf from '@tensorflow/tfjs';

// import * as tf from '@tensorflow/tfjs';
// import { fetch } from '@tensorflow/tfjs-react-native';
// import * as mobilenet from '@tensorflow-models/mobilenet';
// import * as jpeg from 'jpeg-js';

interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  mode: 'results' | 'newStyleImage' | 'newContentImage';
  resultImage?: string;
  styleImage?: string;
  contentImage?: string;
  hasCameraPermission?: boolean;
  cameraType: any;
}

interface ImageInfo {
  uri: string;
  base64?: string;
}

export class WebcamDemo extends React.Component<ScreenProps,ScreenState> {
  private camera?: Camera|null;
  private styler: StyleTranfer;

  constructor(props: ScreenProps) {
    super(props);
    this.state = {
      mode: 'results',
      cameraType: Camera.Constants.Type.front,
      styleImage: 'http://placekitten.com/100/150',
      contentImage: 'http://placekitten.com/100/150',
      resultImage: 'http://placekitten.com/200/400'
    };
    this.styler = new StyleTranfer();
  }

  async componentDidMount() {
    await this.styler.init();
    const { status } = await Permissions.askAsync(Permissions.CAMERA);
    this.setState({ hasCameraPermission: status === 'granted' });
  }

  showResults() {
    this.setState({ mode: 'results' });
  }

  takeStyleImage() {
    this.setState({ mode: 'newStyleImage' });
  }

  takeContentImage() {
    this.setState({ mode: 'newContentImage' });
  }

  flipCamera() {
    const newState = this.state.cameraType === Camera.Constants.Type.back
          ? Camera.Constants.Type.front
          : Camera.Constants.Type.back;
    this.setState({
      cameraType: newState,
    });
  }



  async stylize(contentImage: ImageInfo,
    styleImage: ImageInfo) {
    console.log('YYY- In stylize');
    const contentTensor = await imageUrlToTensor(contentImage.uri,
      contentImage.base64);
    console.log('YYY- Converted content', contentTensor.shape);
    const styleTensor = await imageUrlToTensor(styleImage.uri,
      styleImage.base64);
    console.log('YYY- Converted style', styleTensor.shape);
    const stylizedResult = this.styler.stylize(styleTensor, contentTensor);
    console.log('YYY- finished stylization', stylizedResult.shape);
    const stylizedImage = tensorToImageUrl(stylizedResult);
    console.log('YYY- converted stylization');
    tf.dispose([contentTensor, styleTensor, stylizedResult]);
    console.log('YYY- done with stylize.');
    return stylizedImage;
  }

  async handleCameraCapture() {
    const {mode, styleImage, contentImage, resultImage} = this.state;
    let image = await this.camera!.takePictureAsync();
    image = await resizeImage(image.uri, 150);
    console.log('YYY --- image resized', image.uri, image.width, image.height);
    if(mode === 'newStyleImage') {
      // Compute new stylized result image
      let newResultImage;
      if (contentImage != null) {
        // newResultImage =
        //  `http://placekitten.com/200/${Math.floor(Math.random()*300) + 100}`;
        newResultImage = await this.stylize(
          {uri: contentImage},
          {uri: image.uri, base64: image.base64});
      }

      this.setState({
        styleImage: image.uri,
        mode: 'results',
        resultImage: newResultImage != null ? newResultImage : resultImage,
      });
    } else if (mode === 'newContentImage') {
      let newResultImage;
      // TODO compute result
      if (styleImage != null) {
        // Compute new stylized result image
        // newResultImage =
        //  `http://placekitten.com/200/${Math.floor(Math.random()*300) + 100}`;
        newResultImage = await this.stylize(
          {uri: image.uri, base64: image.base64},
          {uri: styleImage});
      }

      this.setState({
        contentImage: image.uri,
        mode: 'results',
        resultImage: newResultImage != null ? newResultImage : resultImage,
      });
    }
  }

  renderCameraCapture() {
    const {hasCameraPermission} = this.state;

    if (hasCameraPermission === null) {
      return <View />;
    } else if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    }

    return (
      <View  style={{backgroundColor: '#ee2'}}>
        <Camera
          style={styles.camera}
          type={this.state.cameraType}
          ref={ref => { this.camera = ref; }}>
          <View style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.flipCameraBtn}
              onPress={() => {this.flipCamera();}}>
              <Text style={{fontSize: 18, color: 'white'}}>
                  Flip
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.takeImageBtn}
              onPress={() => { this.handleCameraCapture(); }}>
              <Text style={{fontSize: 18, color: 'white'}}>
                  Take
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.cancelBtn}
              onPress={() => {this.showResults(); }}>
              <Text style={{fontSize: 18,  color: 'white'}}>
                  Cancel
              </Text>
            </TouchableOpacity>
          </View>
        </Camera>
        </View>
    );
  }

  renderResults() {
    const {styleImage, contentImage, resultImage} = this.state;
    return (
      <View>
        <View style={styles.resultImageContainer}>
          <Image
            style={styles.resultImage}
            resizeMode='contain'
            source={{uri: resultImage}} />

          <TouchableHighlight
            style={styles.styleImageContainer}
            onPress={() => this.takeStyleImage()}
            underlayColor='white'>
              <View>
                <Image
                  style={{width: 120, height: 150}}
                  source={{uri: styleImage}} />
                  <Text style={styles.centeredText}>Style</Text>
              </View>
          </TouchableHighlight>

          <TouchableHighlight
            style={styles.contentImageContainer}
            onPress={() => this.takeContentImage()}
            underlayColor='white'>
            <View >
              <Image
                style={{width: 120, height: 150}}
                source={{uri: contentImage}} />
                <Text style={styles.centeredText}>Content</Text>
            </View>
          </TouchableHighlight>

        </View>
      </View>
    );
  }

  render() {
    const {mode} = this.state;
    return (
      <Fragment>
        {mode === 'results' ?
              this.renderResults() : this.renderCameraCapture()}
      </Fragment>
    );
  }
}

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24
  },
  centeredText: {
    textAlign: 'center',
    fontSize: 14,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: 'black',
    marginBottom: 6
  },
  camera : {
    width: '100%',
    height: '100%',
    padding:0,
    margin:0,
    backgroundColor: 'transparent',
    zIndex: 1,
  },
  cameraControls: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'space-around',
    position:'absolute',
    bottom: 50,
    zIndex: 100,
    backgroundColor: 'transparent',
    width: '100%',
    height: 75,
  },
  flipCameraBtn: {
    backgroundColor: '#884040',
    width: 75,
    height: 75,
    borderRadius:50,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  takeImageBtn: {
    backgroundColor: '#884040',
    width: 75,
    height: 75,
    borderRadius:50,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cancelBtn: {
    backgroundColor: '#884040',
    width: 75,
    height: 75,
    borderRadius:50,
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  resultImageContainer : {
    width: '100%',
    height: '100%',
    padding:0,
    margin:0,
    backgroundColor: '#fcc',
    zIndex: 1,
  },
  resultImage: {
    width: '100%',
    height: '100%',
  },
  styleImageContainer :{
    position:'absolute',
    bottom: 20,
    left: 20,
    zIndex: 10,
    backgroundColor: '#812',
    borderColor: '#812',
    borderWidth: 2
  },
  contentImageContainer :{
    position:'absolute',
    bottom: 20,
    right: 20,
    zIndex: 10,
    backgroundColor: '#244',
    borderColor: '#244',
    borderWidth: 3
  },

});
