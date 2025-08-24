
const test = require('tap').test;
const chain = require('../../');
const produce = require('../produce.js');

// See issue https://github.com/AndreasMadsen/stack-chain/issues/12 for
// a detailed explaination.

test('formatter works for non-generic (non-safe) toString', (t) => {
  const base = function () {};
  base.toString = base.toString; // sets base.toString to base[[proto]].toString
  Object.setPrototypeOf(base, {}); // sets base[[proto]] = {}

  const error = Object.create(base); // wrap base using prototype chain
  Error.captureStackTrace(error); // prepear error.stack

  t.equal(error.stack.split('\n').length, 11);
  t.end();
});
