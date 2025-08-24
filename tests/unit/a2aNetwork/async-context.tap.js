'use strict';

const tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test('asynchronously propagating state with local-context-domains', (t) => {
  t.plan(2);

  const namespace = createNamespace('namespace');
  t.ok(process.namespaces.namespace, 'namespace has been created');

  namespace.run(() => {
    namespace.set('test', 1337);
    t.equal(namespace.get('test'), 1337, 'namespace is working');
  });
});
