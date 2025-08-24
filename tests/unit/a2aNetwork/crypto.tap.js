'use strict';

const tap             = require('tap')
  , semver          = require('semver')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

let crypto;
try { crypto = require('crypto'); }
catch (err) {}

if (crypto) {
  test('continuation-local state with crypto.randomBytes', (t) => {
    t.plan(1);

    const namespace = createNamespace('namespace');
    namespace.run(() => {
      namespace.set('test', 0xabad1dea);

      t.test('randomBytes', (t) => {
        namespace.run(() => {
          namespace.set('test', 42);
          crypto.randomBytes(100, (err) => {
            if (err) throw err;
            t.equal(namespace.get('test'), 42, 'mutated state was preserved');
            t.end();
          });
        });
      });
    });
  });

  test('continuation-local state with crypto.pseudoRandomBytes', (t) => {
    t.plan(1);

    const namespace = createNamespace('namespace');
    namespace.run(() => {
      namespace.set('test', 0xabad1dea);

      t.test('pseudoRandomBytes', (t) => {
        namespace.run(() => {
          namespace.set('test', 42);
          crypto.pseudoRandomBytes(100, (err) => {
            if (err) throw err;
            t.equal(namespace.get('test'), 42, 'mutated state was preserved');
            t.end();
          });
        });
      });
    });
  });

  test('continuation-local state with crypto.pbkdf2', (t) => {
    t.plan(1);

    const namespace = createNamespace('namespace');
    namespace.run(() => {
      namespace.set('test', 0xabad1dea);

      t.test('pbkdf2', (t) => {
        namespace.run(() => {
          namespace.set('test', 42);
          // this API changed after 0.10.0, and errors if digest is missing after v6
          if (semver.gte(process.version, '0.12.0')) {
            crypto.pbkdf2('s3cr3tz', '451243', 10, 40, 'sha512', (err) => {
              if (err) throw err;
              t.equal(namespace.get('test'), 42, 'mutated state was preserved');
              t.end();
            });
          } else {
            crypto.pbkdf2('s3cr3tz', '451243', 10, 40, (err) => {
              if (err) throw err;
              t.equal(namespace.get('test'), 42, 'mutated state was preserved');
              t.end();
            });
          }
        });
      });
    });
  });
}
