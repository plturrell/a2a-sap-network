'use strict';

const dns             = require('dns')
  , tap             = require('tap')
  , test            = tap.test
  , createNamespace = require('../context.js').createNamespace
  ;

test('continuation-local state with MakeCallback and DNS module', (t) => {
  t.plan(11);

  const namespace = createNamespace('dns');
  namespace.run(() => {
    namespace.set('test', 0xabad1dea);

    t.test('dns.lookup', (t) => {
      namespace.run(() => {
        namespace.set('test', 808);
        t.equal(namespace.get('test'), 808, 'state has been mutated');

        dns.lookup('www.newrelic.com', 4, (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 808,
                  'mutated state has persisted to dns.lookup\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolve', (t) => {
      namespace.run(() => {
        namespace.set('test', 909);
        t.equal(namespace.get('test'), 909, 'state has been mutated');

        dns.resolve('newrelic.com', 'NS', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 909,
                  'mutated state has persisted to dns.resolve\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolve4', (t) => {
      namespace.run(() => {
        namespace.set('test', 303);
        t.equal(namespace.get('test'), 303, 'state has been mutated');

        dns.resolve4('www.newrelic.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 303,
                  'mutated state has persisted to dns.resolve4\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolve6', (t) => {
      namespace.run(() => {
        namespace.set('test', 101);
        t.equal(namespace.get('test'), 101, 'state has been mutated');

        dns.resolve6('google.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 101,
                  'mutated state has persisted to dns.resolve6\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveCname', (t) => {
      namespace.run(() => {
        namespace.set('test', 212);
        t.equal(namespace.get('test'), 212, 'state has been mutated');

        dns.resolveCname('mail.newrelic.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 212,
                  'mutated state has persisted to dns.resolveCname\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveMx', (t) => {
      namespace.run(() => {
        namespace.set('test', 707);
        t.equal(namespace.get('test'), 707, 'state has been mutated');

        dns.resolveMx('newrelic.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 707,
                  'mutated state has persisted to dns.resolveMx\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveNs', (t) => {
      namespace.run(() => {
        namespace.set('test', 717);
        t.equal(namespace.get('test'), 717, 'state has been mutated');

        dns.resolveNs('newrelic.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 717,
                  'mutated state has persisted to dns.resolveNs\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveTxt', (t) => {
      namespace.run(() => {
        namespace.set('test', 2020);
        t.equal(namespace.get('test'), 2020, 'state has been mutated');

        dns.resolveTxt('newrelic.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 2020,
                  'mutated state has persisted to dns.resolveTxt\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveSrv', (t) => {
      namespace.run(() => {
        namespace.set('test', 9000);
        t.equal(namespace.get('test'), 9000, 'state has been mutated');

        dns.resolveSrv('_xmpp-server._tcp.google.com', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 9000,
                  'mutated state has persisted to dns.resolveSrv\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.resolveNaptr', (t) => {
      // dns.resolveNaptr only in Node > 0.9.x
      if (!dns.resolveNaptr) return t.end();

      namespace.run(() => {
        namespace.set('test', 'Polysix');
        t.equal(namespace.get('test'), 'Polysix', 'state has been mutated');

        dns.resolveNaptr('columbia.edu', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 'Polysix',
                  'mutated state has persisted to dns.resolveNaptr\'s callback');

          t.end();
        });
      });
    });

    t.test('dns.reverse', (t) => {
      namespace.run(() => {
        namespace.set('test', 1000);
        t.equal(namespace.get('test'), 1000, 'state has been mutated');

        dns.reverse('204.93.223.144', (err, addresses) => {
          t.notOk(err, 'lookup succeeded');
          t.ok(addresses.length > 0, 'some results were found');

          t.equal(namespace.get('test'), 1000,
                  'mutated state has persisted to dns.reverse\'s callback');

          t.end();
        });
      });
    });
  });
});
