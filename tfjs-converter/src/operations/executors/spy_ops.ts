export type RecursiveSpy<T> = T extends Function ? jasmine.Spy : {
  [K in keyof T]: RecursiveSpy<T[K]>
};

export function spyOnAllFunctions<T>(obj: T): RecursiveSpy<T> {
  return Object.fromEntries(
    Object.entries(obj).map(([key, val]) => {
      if (val instanceof Function) {
        return [key, jasmine.createSpy(`${key} spy`, val).and.callThrough()];
      } else if (val instanceof Array) {
        return [key, val];
      } else if (val instanceof Object) {
        return [key, spyOnAllFunctions(val)];
      }
      return [key, val];
    })) as RecursiveSpy<T>;
}
