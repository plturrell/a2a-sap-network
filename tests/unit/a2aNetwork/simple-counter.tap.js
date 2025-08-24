const test = require('tap').test;

test('asyncListeners work as expected with process.nextTick', (t) => {
  t.plan(4);

  if (!process.addAsyncListener) require('../index.js');

  let active
    , cntr   = 0
    ;

  process.addAsyncListener(
    {
      create : function () { return { val : ++cntr }; },
      before : function (context, data) { active = data.val; },
      after  : function () { active = null; }
    }
  );

  process.nextTick(() => {
    t.equal(active, 1);
    process.nextTick(() => { t.equal(active, 3); });
  });

  process.nextTick(() => {
    t.equal(active, 2);
    process.nextTick(() => { t.equal(active, 4); });
  });
});
