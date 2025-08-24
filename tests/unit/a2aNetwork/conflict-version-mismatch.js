
const test = require('tap').test;

global._stackChain = { version: 'unlikely' };

test('diffrent version but copies', (t) => {
  try {
    require('../../');
  } catch (e) {
    t.equal(e.message, 'Conflicting version of stack-chain found');
    t.end();
  }
});
