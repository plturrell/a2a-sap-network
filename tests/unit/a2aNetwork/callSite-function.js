
const test = require('tap').test;
const chain = require('../../');
const produce = require('../produce.js');

Error.stackTraceLimit = Infinity;

test('stack extend part', (t) => {
  const extend = function (error, frames) {
    frames.splice(1, 0, 'EXTEND', 'FILTER ME');
    return frames;
  };

  const filter = function (error, frames) {
    return frames.filter((callSite) => {
      return callSite !== 'FILTER ME';
    });
  };

  const callSites = function (level, options) {
    const limit = Error.stackTraceLimit;
    let callSites;
    produce.deepStack(0, level, () => {
      Error.stackTraceLimit = level;
      callSites = chain.callSite(options);
      Error.stackTraceLimit = limit;
    });

    return callSites.slice(1, Infinity);
  };

  t.test('callSite method matches simple case property length', (t) => {
    const method = chain.callSite();
    const propery = (new Error()).callSite.original;
    t.strictEqual(method.length, propery.length);

    // The other stuff still works
    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('pretest: toString of callSites array', (t) => {
    t.equal(produce.convert(callSites(3)), produce.fake([
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('callSite with extend', (t) => {
    chain.extend.attach(extend);
    const textA = produce.convert(callSites(3, { extend: true }));
    const textB = produce.convert(callSites(3));
    chain.extend.deattach(extend);

    t.equal(textA, produce.fake([
      '    at EXTEND',
      '    at FILTER ME',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.equal(textB, produce.fake([
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('callSite with extend and filter', (t) => {
    chain.extend.attach(extend);
    chain.filter.attach(filter);
    const textA = produce.convert(callSites(3, { extend: true, filter: true }));
    const textB = produce.convert(callSites(3, { filter: true }));
    chain.filter.deattach(filter);
    chain.extend.deattach(extend);

    t.equal(textA, produce.fake([
      '    at EXTEND',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.equal(textB, produce.fake([
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('callSite with extend and filter and slice', (t) => {
    chain.extend.attach(extend);
    chain.filter.attach(filter);
    const textA = produce.convert(callSites(3, { extend: true, filter: true, slice: 1 }));
    const textB = produce.convert(callSites(3, { slice: 1 }));
    chain.filter.deattach(filter);
    chain.extend.deattach(extend);

    t.equal(textA, produce.fake([
      '    at EXTEND',
      '    at deepStack ({where}:7:5)'
    ]));

    t.equal(textB, produce.fake([
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.end();
});
