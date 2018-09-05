import {renderConfusionMatrix} from '../render/confusion_matrix';
import {getDrawArea} from '../render/render_utils';
import {renderTable} from '../render/table';
import {ConfusionMatrixData, Drawable} from '../types';

/**
 * Renders a per class accuracy table for classification task evaluation
 *
 * @param container A `{name: string, tab?: string}` object specifying which
 * surface to render to.
 * @param classAccuracy A `{accuracy: number[], count: number[]}` object with
 * the accuracy data. See metrics.perClassAccuracy for details on how to
 * generate this object.
 * @param classLabels An array of string labels for the classes in
 * `classAccuracy`. Optional.
 */
export async function perClassAccuracy(
    container: Drawable, classAccuracy: {accuracy: number[], count: number[]},
    classLabels?: string[]) {
  const drawArea = getDrawArea(container);

  const headers = [
    'Class',
    'Accuracy',
    '# Samples',
  ];
  const values: Array<Array<(string | number)>> = [];

  for (let i = 0; i < classAccuracy.accuracy.length; i++) {
    const label = classLabels ? classLabels[i] : i.toString();
    const acc = classAccuracy.accuracy[i];
    const count = classAccuracy.count[i];
    values.push([label, acc, count]);
  }

  return renderTable({headers, values}, drawArea);
}

/**
 * Renders a confusion matrix for classification task evaluation
 *
 * @param container A `{name: string, tab?: string}` object specifying which
 * surface to render to.
 * @param confusionMatrix A nested array of numbers with the confusion matrix
 * values. See metrics.confusionMatrix for details on how to generate this.
 * @param classLabels An array of string labels for the classes in
 * `confusionMatrix`. Optional.
 */
export async function confusionMatrix(
    container: Drawable, confusionMatrix: number[][], classLabels?: string[]) {
  const drawArea = getDrawArea(container);

  const confusionMatrixData: ConfusionMatrixData = {
    values: confusionMatrix,
    labels: classLabels,
  };

  return renderConfusionMatrix(confusionMatrixData, drawArea, {
    height: 450,
  });
}
