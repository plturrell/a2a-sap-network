const test = require('tap').test
  , cls  = require('../context.js')
  ;

test('minimized test case that caused #6011 patch to fail', (t) => {
  t.plan(3);

  console.log('+');
  // when the flaw was in the patch, commenting out this line would fix things:
  process.nextTick(() => { console.log('!'); });

  const n = cls.createNamespace('test');
  t.ok(!n.get('state'), 'state should not yet be visible');

  n.run(() => {
    n.set('state', true);
    t.ok(n.get('state'), 'state should be visible');

    process.nextTick(() => {
      t.ok(n.get('state'), 'state should be visible');
    });
  });
});
