'use strict';

const tap = require('tap');
const test = tap.test;
const sinon = require('sinon');
const shimmer = require('../index.js');

let outsider = 0;
function counter () { return ++outsider; }
function anticounter () { return --outsider; }

const generator = {
  inc: counter
};
Object.defineProperty(generator, 'dec', {
  value: anticounter,
  writable: true,
  configurable: true,
  enumerable: false
});

test('should wrap safely', (t) => {
  t.plan(12);

  t.equal(counter, generator.inc, 'method is mapped to function');
  t.doesNotThrow(() => { generator.inc(); }, 'original function works');
  t.equal(1, outsider, 'calls have side effects');

  let count = 0;
  function wrapper (original, name) {
    t.equal(name, 'inc');
    return function () {
      count++;
      const returned = original.apply(this, arguments);
      count++;
      return returned;
    };
  }
  shimmer.wrap(generator, 'inc', wrapper);

  t.ok(generator.inc.__wrapped, 'function tells us it\'s wrapped');
  t.equal(generator.inc.__original, counter, 'original function is available');
  t.doesNotThrow(() => { generator.inc(); }, 'wrapping works');
  t.equal(2, count, 'both pre and post increments should have happened');
  t.equal(2, outsider, 'original function has still been called');
  t.ok(generator.propertyIsEnumerable('inc'),
    'wrapped enumerable property is still enumerable');
  t.equal(Object.keys(generator.inc).length, 0,
    'wrapped object has no additional properties');

  shimmer.wrap(generator, 'dec', (original) => {
    return function () {
      return original.apply(this, arguments);
    };
  });

  t.ok(!generator.propertyIsEnumerable('dec'),
    'wrapped unenumerable property is still unenumerable');
});

test('wrap called with no arguments', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('no original function undefined to wrap')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.wrap();
  }, 'wrapping with no arguments doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with module but nothing else', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('no original function undefined to wrap')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.wrap(generator);
  }, 'wrapping with only 1 argument doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with original but no wrapper', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .twice();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.wrap(generator, 'inc');
  }, 'wrapping with only original method doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with non-function original', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('original object and wrapper must be functions')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.wrap({ orange: 'slices' }, 'orange', () => {});
  }, 'wrapping non-function original doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with non-function wrapper', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withArgs('original object and wrapper must be functions')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.wrap({ orange: function () {} }, 'orange', 'hamchunx');
  }, 'wrapping with non-function wrapper doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});
