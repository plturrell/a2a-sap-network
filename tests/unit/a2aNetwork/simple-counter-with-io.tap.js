const test = require('tap').test;

if (!global.setImmediate) global.setImmediate = setTimeout;

test('asyncListeners work as expected with process.nextTick', (t) => {
  t.plan(1);

  if (!process.addAsyncListener) require('../index.js');

  console.log('+');
  // comment out this line to get the expected result:
  setImmediate(() => { console.log('!'); });

  let counter = 1;
  let current;
  process.addAsyncListener(
    {
      create : function listener() { return counter++; },
      before : function (_, domain) { current = domain; },
      after  : function () { current = null; }
    }
  );

  setImmediate(() => { t.equal(current, 1); });
  // uncomment this line to get the expected result:
  // process.removeAsyncListener(id);
});

