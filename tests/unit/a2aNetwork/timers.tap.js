'use strict';

const tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test('continuation-local state with timers', (t) => {
  t.plan(4);

  const namespace = createNamespace('namespace');
  namespace.run(() => {
    namespace.set('test', 0xabad1dea);

    t.test('process.nextTick', (t) => {
      namespace.run(() => {
        namespace.set('test', 31337);
        t.equal(namespace.get('test'), 31337, 'state has been mutated');

        process.nextTick(() => {
          t.equal(namespace.get('test'), 31337,
                  'mutated state has persisted to process.nextTick\'s callback');

          t.end();
        });
      });
    });

    t.test('setImmediate', (t) => {
      // setImmediate only in Node > 0.9.x
      if (!global.setImmediate) return t.end();

      namespace.run(() => {
        namespace.set('test', 999);
        t.equal(namespace.get('test'), 999, 'state has been mutated');

        setImmediate(() => {
          t.equal(namespace.get('test'), 999,
                  'mutated state has persisted to setImmediate\'s callback');

          t.end();
        });
      });
    });

    t.test('setTimeout', (t) => {
      namespace.run(() => {
        namespace.set('test', 54321);
        t.equal(namespace.get('test'), 54321, 'state has been mutated');

        setTimeout(() => {
          t.equal(namespace.get('test'), 54321,
                  'mutated state has persisted to setTimeout\'s callback');

          t.end();
        });
      });
    });

    t.test('setInterval', (t) => {
      namespace.run(() => {
        namespace.set('test', 10101);
        t.equal(namespace.get('test'), 10101,
                'continuation-local state has been mutated');

        var ref = setInterval(() => {
          t.equal(namespace.get('test'), 10101,
                  'mutated state has persisted to setInterval\'s callback');

          clearInterval(ref);
          t.end();
        }, 20);
      });
    });
  });
});
