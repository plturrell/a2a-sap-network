
const test = require('tap').test;

const first = global._stackChain = { version: require('../../package.json').version };
const chain = require('../../');

test('same version but copies', (t) => {
  t.strictEqual(chain, first);
  t.end();
});
