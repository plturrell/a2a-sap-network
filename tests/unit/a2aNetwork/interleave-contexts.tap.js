'use strict';

const cls  = require('../context.js')
  , test = require('tap').test
  ;

function cleanNamespace(name){
  if (cls.getNamespace(name)) cls.destroyNamespace(name);
  return cls.createNamespace(name);
}

test('interleaved contexts', (t) => {
  t.plan(3);

  t.test('interleaving with run', (t) => {
    t.plan(2);

    const ns = cleanNamespace('test');

    const ctx = ns.createContext();

    ns.enter(ctx);
    ns.run(() => {
      t.equal(ns._set.length, 2, '2 contexts in the active set');
      t.doesNotThrow(() => { ns.exit(ctx); });
    });
  });

  t.test('entering and exiting staggered', (t) => {
    t.plan(4);

    const ns = cleanNamespace('test');

    const ctx1 = ns.createContext();
    const ctx2 = ns.createContext();

    t.doesNotThrow(() => { ns.enter(ctx1); });
    t.doesNotThrow(() => { ns.enter(ctx2); });

    t.doesNotThrow(() => { ns.exit(ctx1); });
    t.doesNotThrow(() => { ns.exit(ctx2); });
  });

  t.test('creating, entering and exiting staggered', (t) => {
    t.plan(4);

    const ns = cleanNamespace('test');

    const ctx1 = ns.createContext();
    t.doesNotThrow(() => { ns.enter(ctx1); });

    const ctx2 = ns.createContext();
    t.doesNotThrow(() => { ns.enter(ctx2); });

    t.doesNotThrow(() => { ns.exit(ctx1); });
    t.doesNotThrow(() => { ns.exit(ctx2); });
  });
});
