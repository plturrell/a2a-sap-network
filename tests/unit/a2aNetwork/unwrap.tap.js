'use strict';

const tap = require('tap');
const test = tap.test;
const sinon = require('sinon');
const shimmer = require('../index.js');

let outsider = 0;
function counter () { return ++outsider; }

const generator = {
  inc: counter
};

test('should unwrap safely', (t) => {
  t.plan(9);

  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.doesNotThrow(() => { generator.inc(); });
  t.equal(1, outsider, 'calls have side effects');

  function wrapper (original) {
    return function () {
      return original.apply(this, arguments);
    };
  }
  shimmer.wrap(generator, 'inc', wrapper);

  t.doesNotEqual(counter, generator.inc, 'function should be wrapped');

  t.doesNotThrow(() => { generator.inc(); });
  t.equal(2, outsider, 'original function has still been called');

  shimmer.unwrap(generator, 'inc');
  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.doesNotThrow(() => { generator.inc(); });
  t.equal(3, outsider, 'original function has still been called');
});

test('shouldn\'t throw on double unwrapping', (t) => {
  t.plan(6);

  t.equal(counter, generator.inc, 'basic function equality testing should work');

  const mock = sinon.expectation
    .create('logger')
    .withArgs('no original to unwrap to -- ' +
      'has inc already been unwrapped?')
    .once();
  shimmer({ logger: mock });

  function wrapper (original) {
    return function () {
      return original.apply(this, arguments);
    };
  }
  shimmer.wrap(generator, 'inc', wrapper);

  t.doesNotEqual(counter, generator.inc, 'function should be wrapped');

  shimmer.unwrap(generator, 'inc');
  t.equal(counter, generator.inc, 'basic function equality testing should work');

  t.doesNotThrow(() => { shimmer.unwrap(generator, 'inc'); },
    'should double unwrap without issue');
  t.equal(counter, generator.inc, 'function is unchanged after unwrapping');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('unwrap called with no arguments', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .twice();
  shimmer({ logger: mock });

  t.doesNotThrow(() => { shimmer.unwrap(); }, 'should log instead of throwing');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('unwrap called with module but no name', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .twice();
  shimmer({ logger: mock });

  t.doesNotThrow(() => { shimmer.unwrap({}); }, 'should log instead of throwing');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});
