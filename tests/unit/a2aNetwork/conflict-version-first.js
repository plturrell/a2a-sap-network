
const test = require('tap').test;
const chain = require('../../');

test('no other copy', (t) => {
  t.strictEqual(global._stackChain, chain);
  t.end();
});
