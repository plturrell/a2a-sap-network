/* eslint-env mocha */

const assert = require('assert');
const appendField = require('../');
const testData = require('testdata-w3c-json-form');

describe('Append Field', () => {
  for (var test of testData) {
    it(`handles ${  test.name}`, () => {
      const store = Object.create(null);

      for (const field of test.fields) {
        appendField(store, field.key, field.value);
      }

      assert.deepEqual(store, test.expected);
    });
  }
});
