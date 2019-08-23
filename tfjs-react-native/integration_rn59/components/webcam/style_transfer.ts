import * as tf from '@tensorflow/tfjs';

export class StyleTranfer {
  private styleNet?: tf.GraphModel;
  private transformNet?: tf.GraphModel;

  constructor() {}

  async init() {
    await Promise.all([this.loadStyleModel(), this.loadTransformerModel()]);
    // this.testStylization();
  }

  async loadStyleModel() {
    if (this.styleNet == null) {
      this.styleNet = await tf.loadGraphModel(
          // tslint:disable-next-line: max-line-length
          'https://cdn.jsdelivr.net/gh/reiinakano/arbitrary-image-stylization-tfjs@master/saved_model_style_js/model.json');
      console.log('yyy stylenet loaded');
    }
  }

  async loadTransformerModel() {
    if (this.transformNet == null) {
      this.transformNet = await tf.loadGraphModel(
          // tslint:disable-next-line: max-line-length
          'https://cdn.jsdelivr.net/gh/reiinakano/arbitrary-image-stylization-tfjs@master/saved_model_transformer_separable_js/model.json');
      console.log('yyy transformnet loaded');
    }
  }

  testStylization() {
    const styleT = tf.randomNormal([256, 256, 3]) as tf.Tensor3D;
    const contentT = tf.randomNormal([256, 256, 3]) as tf.Tensor3D;

    const res = this.stylize(styleT, contentT);
    const resData = Array.from(res.dataSync());
    console.log('yyy testStylization result', resData.slice(0, 10));
    tf.dispose([styleT, contentT, res]);
  }

  /**
   * This function returns style bottleneck features for
   * the given image.
   *
   * @param style Style image to get 100D bottleneck features for
   */
  private predictStyleParameters(styleImage: tf.Tensor3D): tf.Tensor4D {
    return tf.tidy(() => {
      return this.styleNet!.predict(
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
    // console.log(
    //     'yyy bottleneck', bottleneck.shape, bottleneck.dataSync().length,
    //     Array.from(bottleneck.dataSync()).slice(0, 10));
    return tf.tidy(() => {
      const input = contentImage.toFloat().div(tf.scalar(255)).expandDims();
      // console.log(
      //     'yyy input', input.shape, input.dataSync().length,
      //     Array.from(input.dataSync()).slice(0, 10));
      const image: tf.Tensor4D =
          this.transformNet!.predict([input, bottleneck]) as tf.Tensor4D;
      // console.log('yyy predict output');
      // console.log(
      //     'yyy predict output', image.dataSync().length,
      //     Array.from(image.dataSync()).slice(0, 10));
      return image.mul(255).squeeze();
    });
  }

  public stylize(
      styleImage: tf.Tensor3D, contentImage: tf.Tensor3D,
      strength?: number): tf.Tensor3D {
    const start = Date.now();
    // console.log('yyy in styletransfer.stylize');
    // console.log(
    //     'yyy input', styleImage.shape, contentImage.shape,
    //     Array.from(styleImage.dataSync().slice(0, 10)),
    //     Array.from(contentImage.dataSync().slice(0, 10)));
    let styleRepresentation = this.predictStyleParameters(styleImage);
    // console.log('stylerepresentation');
    // styleRepresentation.print();

    // console.log('contentImage');
    // contentImage.print();
    // if (strength != null) {
    //   styleRepresentation = tf.tidy(
    //       () => styleRepresentation.mul(tf.scalar(strength))
    //                 .add(this.predictStyleParameters(contentImage)
    //                          .mul(tf.scalar(1.0 - strength))));
    // }

    const stylized = this.produceStylized(contentImage, styleRepresentation);

    const transposed = stylized.transpose([1, 0, 2]);
    // console.log(
    //     'yyy stylized',
    //     Array.from(stylized.dataSync().slice(0, 10)),
    // );
    // console.log(
    //     'yyy transposed', Array.from(transposed.dataSync().slice(0, 10)));

    tf.dispose([styleRepresentation, transposed]);
    const end = Date.now();
    console.log('yyy stylization complete', end - start);
    return stylized;
  }
}
