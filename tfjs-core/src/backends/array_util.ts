/**
 * Inserts a value into a sorted array. This method allows duplicate, meaning it
 * allows inserting duplicate value, in which case, the element will be inserted
 * at the lowest index of the value.
 * @param arr The array to modify.
 * @param element The element to insert.
 * @param comparator Optional. If no comparator is specified, elements are
 * compared using array_util.defaultComparator, which is suitable for Strings
 * and Numbers in ascending arrays. If the array contains multiple instances of
 * the target value, the left-most instance will be returned. To provide a
 * comparator, it should take 2 arguments to compare and return a negative,
 * zero, or a positive number.
 */
export function binaryInsert<T>(
    arr: T[], element: T, comparator?: (a: T, b: T) => number) {
  const index = binarySearch(arr, element, comparator);
  const insertionPoint = index < 0 ? -(index + 1) : index;
  arr.splice(insertionPoint, 0, element);
}

/**
 * Searches the array for the target using binary search, returns the index
 * of the found element, or position to insert if element not found. If no
 * comparator is specified, elements are compared using array_
 * util.defaultComparator, which is suitable for Strings and Numbers in
 * ascending arrays. If the array contains multiple instances of the target
 * value, the left-most instance will be returned.
 * @param arr The array to be searched in.
 * @param target The target to be searched for.
 * @param comparator Should take 2 arguments to compare and return a negative,
 *    zero, or a positive number.
 * @return Lowest index of the target value if found, otherwise the insertion
 *    point where the target should be inserted, in the form of
 *    (-insertionPoint - 1).
 */
export function binarySearch<T>(
    arr: T[], target: T, comparator?: (a: T, b: T) => number) {
  return binarySearch_(arr, target, comparator || defaultComparator);
}

/**
 * Compares its two arguments for order.
 * @param a The first element to be compared.
 * @param b The second element to be compared.
 * @return A negative number, zero, or a positive number as the first
 *     argument is less than, equal to, or greater than the second.
 */
function defaultComparator<T>(a: T, b: T): number {
  return a > b ? 1 : a < b ? -1 : 0;
}

function binarySearch_<T>(
    arr: T[], target: T, comparator: (a: T, b: T) => number) {
  let left = 0;
  let right = arr.length;
  let middle = 0;
  let found = false;
  while (left < right) {
    middle = left + ((right - left) >>> 1);
    const compareResult = comparator(target, arr[middle]);
    if (compareResult > 0) {
      left = middle + 1;
    } else {
      right = middle;
      // If compareResult is 0, the value is found. We record it is found,
      // and then keep looking because there may be duplicate.
      found = !compareResult;
    }
  }

  return found ? left : -left - 1;
}
