import * as binding from '../.';

describe('Exposes TF_DataType enum values', () => {
  it('contains TF_FLOAT', () => {
    expect(binding.TF_FLOAT).toEqual(1);
  });
  it('contains TF_INT32', () => {
    expect(binding.TF_INT32).toEqual(3);
  });
  it('contains TF_BOOL', () => {
    expect(binding.TF_BOOL).toEqual(10);
  });
});

describe('Exposes TF_AttrType enum values', () => {
  it('contains TF_ATTR_STRING', () => {
    expect(binding.TF_ATTR_STRING).toEqual(0);
  });
  it('contains TF_ATTR_INT', () => {
    expect(binding.TF_ATTR_INT).toEqual(1);
  });
  it('contains TF_ATTR_FLOAT', () => {
    expect(binding.TF_ATTR_FLOAT).toEqual(2);
  });
  it('contains TF_ATTR_BOOL', () => {
    expect(binding.TF_ATTR_BOOL).toEqual(3);
  });
  it('contains TF_ATTR_TYPE', () => {
    expect(binding.TF_ATTR_TYPE).toEqual(4);
  });
  it('contains TF_ATTR_SHAPE', () => {
    expect(binding.TF_ATTR_SHAPE).toEqual(5);
  });
  it('contains TF_ATTR_TENSOR', () => {
    expect(binding.TF_ATTR_TENSOR).toEqual(6);
  });
  it('contains TF_ATTR_PLACEHOLDER', () => {
    expect(binding.TF_ATTR_PLACEHOLDER).toEqual(7);
  });
  it('contains TF_ATTR_FUNC', () => {
    expect(binding.TF_ATTR_FUNC).toEqual(8);
  });
});

describe('Exposes TF Version', () => {
  it('contains a version string', () => {
    expect(binding.TF_Version).toBeDefined();
  });
});
