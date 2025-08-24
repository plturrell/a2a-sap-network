'use strict';

const tap = require('tap');
const test = tap.test;

const context = require('../context.js');

test('namespace management', (t) => {
  t.plan(8);

  t.throws(() => { context.createNamespace(); }, 'name is required');

  let namespace = context.createNamespace('test');
  t.ok(namespace, 'namespace is returned upon creation');

  t.equal(context.getNamespace('test'), namespace, 'namespace lookup works');

  t.doesNotThrow(() => { context.reset(); }, 'allows resetting namespaces');

  t.equal(Object.keys(process.namespaces).length, 0, 'namespaces have been reset');

  namespace = context.createNamespace('another');
  t.ok(process.namespaces.another, 'namespace is available from global');

  t.doesNotThrow(() => { context.destroyNamespace('another'); },
                 'destroying works');

  t.notOk(process.namespaces.another, 'namespace has been removed');
});
