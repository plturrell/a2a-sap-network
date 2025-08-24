'use strict';

const tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

const zlib = require('zlib');

test('continuation-local state with zlib', (t) => {
  t.plan(1);

  const namespace = createNamespace('namespace');
  namespace.run(() => {
    namespace.set('test', 0xabad1dea);

    t.test('deflate', (t) => {
      namespace.run(() => {
        namespace.set('test', 42);
        zlib.deflate(new Buffer('Goodbye World'), (err) => {
          if (err) throw err;
          t.equal(namespace.get('test'), 42, 'mutated state was preserved');
          t.end();
        });
      });
    });
  });
});
