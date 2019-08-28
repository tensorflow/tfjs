import * as tf from '@tensorflow/tfjs';

export class StyleTranfer {
  private styleNet?: tf.GraphModel;
  private transformNet?: tf.GraphModel;

  constructor() {}

  async init() {
    await Promise.all([this.loadStyleModel(), this.loadTransformerModel()]);
    await this.warmup();
  }

  async loadStyleModel() {
    if (this.styleNet == null) {
      this.styleNet = await tf.loadGraphModel(
          // tslint:disable-next-line: max-line-length
          'https://cdn.jsdelivr.net/gh/reiinakano/arbitrary-image-stylization-tfjs@master/saved_model_style_js/model.json');
      console.log('stylenet loaded');
    }
  }

  async loadTransformerModel() {
    if (this.transformNet == null) {
      this.transformNet = await tf.loadGraphModel(
          // tslint:disable-next-line: max-line-length
          'https://cdn.jsdelivr.net/gh/reiinakano/arbitrary-image-stylization-tfjs@master/saved_model_transformer_separable_js/model.json');
      console.log('transformnet loaded');
    }
  }

  async warmup() {
    // Also warmup
    const input = tf.randomNormal([320, 240, 3]) as tf.Tensor3D;
    const res = this.stylize(input, input);
    await res.data();
    tf.dispose([input, res]);
  }

  /**
   * This function returns style bottleneck features for
   * the given image.
   *
   * @param style Style image to get 100D bottleneck features for
   */
  private predictStyleParameters(styleImage: tf.Tensor3D): tf.Tensor4D {
    return tf.tidy(() => {
      if (this.styleNet == null) {
        throw new Error('Stylenet not loaded');
      }
      return this.styleNet.predict(
          styleImage.toFloat().div(tf.scalar(255)).expandDims());
    }) as tf.Tensor4D;
  }

  /**
   * This function stylizes the content image given the bottleneck
   * features. It returns a tf.Tensor3D containing the stylized image.
   *
   * @param content Content image to stylize
   * @param bottleneck Bottleneck features for the style to use
   */
  private produceStylized(contentImage: tf.Tensor3D, bottleneck: tf.Tensor4D):
      tf.Tensor3D {
    return tf.tidy(() => {
      if (this.transformNet == null) {
        throw new Error('Transformnet not loaded');
      }
      const input = contentImage.toFloat().div(tf.scalar(255)).expandDims();
      const image: tf.Tensor4D =
          this.transformNet.predict([input, bottleneck]) as tf.Tensor4D;
      return image.mul(255).squeeze();
    });
  }

  public stylize(styleImage: tf.Tensor3D, contentImage: tf.Tensor3D):
      tf.Tensor3D {
    const start = Date.now();
    console.log(styleImage.shape, contentImage.shape);
    const styleRepresentation = this.predictStyleParameters(styleImage);
    const stylized = this.produceStylized(contentImage, styleRepresentation);
    tf.dispose([styleRepresentation]);
    const end = Date.now();
    console.log('stylization scheduled', end - start);
    return stylized;
  }
}
