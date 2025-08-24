'use strict';

const tap = require('tap');
const test = tap.test;
const sinon = require('sinon');
const shimmer = require('../index.js');

let outsider = 0;
function counter () { return ++outsider; }
function anticounter () { return --outsider; }

const generator = {
  inc: counter,
  dec: anticounter
};

test('should unwrap safely', (t) => {
  t.plan(18);

  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.equal(anticounter, generator.dec, 'basic function equality testing should work');
  t.doesNotThrow(() => { generator.inc(); });
  t.equal(1, outsider, 'calls have side effects');
  t.doesNotThrow(() => { generator.dec(); });
  t.equal(0, outsider, 'calls have side effects');

  function wrapper (original) {
    return function () {
      return original.apply(this, arguments);
    };
  }
  shimmer.massWrap(generator, ['inc', 'dec'], wrapper);

  t.doesNotEqual(counter, generator.inc, 'function should be wrapped');
  t.doesNotEqual(anticounter, generator.dec, 'function should be wrapped');

  t.doesNotThrow(() => { generator.inc(); });
  t.equal(1, outsider, 'original function has still been called');
  t.doesNotThrow(() => { generator.dec(); });
  t.equal(0, outsider, 'original function has still been called');

  shimmer.massUnwrap(generator, ['inc', 'dec']);
  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.equal(anticounter, generator.dec, 'basic function equality testing should work');

  t.doesNotThrow(() => { generator.inc(); });
  t.equal(1, outsider, 'original function has still been called');
  t.doesNotThrow(() => { generator.dec(); });
  t.equal(0, outsider, 'original function has still been called');
});

test('shouldn\'t throw on double unwrapping', (t) => {
  t.plan(10);

  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.equal(anticounter, generator.dec, 'basic function equality testing should work');

  const mock = sinon.stub();
  shimmer({ logger: mock });

  function wrapper (original) {
    return function () {
      return original.apply(this, arguments);
    };
  }
  shimmer.wrap(generator, 'inc', wrapper);
  shimmer.wrap(generator, 'dec', wrapper);

  t.doesNotEqual(counter, generator.inc, 'function should be wrapped');
  t.doesNotEqual(anticounter, generator.dec, 'function should be wrapped');

  shimmer.massUnwrap(generator, ['inc', 'dec']);
  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.equal(anticounter, generator.dec, 'basic function equality testing should work');

  t.doesNotThrow(() => { shimmer.massUnwrap(generator, ['inc', 'dec']); },
    'should double unwrap without issue');
  t.equal(counter, generator.inc, 'function is unchanged after unwrapping');
  t.equal(anticounter, generator.dec, 'function is unchanged after unwrapping');

  t.doesNotThrow(() => {
    sinon.assert.calledWith(mock, 'no original to unwrap to -- ' +
      'has inc already been unwrapped?');
    sinon.assert.calledWith(mock, 'no original to unwrap to -- ' +
      'has dec already been unwrapped?');
    sinon.assert.calledTwice(mock);
  }, 'logger was called with the expected message');
});

test('massUnwrap called with no arguments', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .twice();
  shimmer({ logger: mock });

  t.doesNotThrow(() => { shimmer.massUnwrap(); }, 'should log instead of throwing');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('massUnwrap called with module but nothing else', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('must provide one or more functions to unwrap on modules')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.massUnwrap(generator);
  }, 'wrapping with only 1 argument doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});
