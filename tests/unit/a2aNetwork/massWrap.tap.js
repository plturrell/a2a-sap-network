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

const arrow = {
  in: counter,
  out: anticounter
};

const nester = {
  in: counter,
  out: anticounter
};

test('should wrap multiple functions safely', (t) => {
  t.plan(9);

  t.equal(counter, generator.inc, 'basic function equality testing should work');
  t.equal(anticounter, generator.dec, 'basic function equality testing should work');
  t.doesNotThrow(() => { generator.inc(); });
  t.doesNotThrow(() => { generator.dec(); });
  t.equal(0, outsider, 'calls have side effects');

  let count = 0;
  function wrapper (original) {
    return function () {
      count++;
      const returned = original.apply(this, arguments);
      count++;
      return returned;
    };
  }
  shimmer.massWrap(generator, ['inc', 'dec'], wrapper);

  t.doesNotThrow(() => { generator.inc(); });
  t.doesNotThrow(() => { generator.dec(); });
  t.equal(4, count, 'both pre and post increments should have happened');
  t.equal(0, outsider, 'original function has still been called');
});

test('should wrap multiple functions on multiple modules safely', (t) => {
  t.plan(15);

  t.equal(counter, arrow.in, 'basic function equality testing should work');
  t.equal(counter, nester.in, 'basic function equality testing should work');
  t.equal(anticounter, arrow.out, 'basic function equality testing should work');
  t.equal(anticounter, nester.out, 'basic function equality testing should work');

  t.doesNotThrow(() => { arrow.in(); });
  t.doesNotThrow(() => { nester.in(); });
  t.doesNotThrow(() => { arrow.out(); });
  t.doesNotThrow(() => { nester.out(); });

  t.equal(0, outsider, 'calls have side effects');

  let count = 0;
  function wrapper (original) {
    return function () {
      count++;
      const returned = original.apply(this, arguments);
      count++;
      return returned;
    };
  }
  shimmer.massWrap([arrow, nester], ['in', 'out'], wrapper);

  t.doesNotThrow(() => { arrow.in(); });
  t.doesNotThrow(() => { arrow.out(); });
  t.doesNotThrow(() => { nester.in(); });
  t.doesNotThrow(() => { nester.out(); });

  t.equal(8, count, 'both pre and post increments should have happened');
  t.equal(0, outsider, 'original function has still been called');
});

test('wrap called with no arguments', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .twice();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.massWrap();
  }, 'wrapping with no arguments doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with module but nothing else', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('must provide one or more functions to wrap on modules')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.massWrap(generator);
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
    shimmer.massWrap(generator, ['inc']);
  }, 'wrapping with only original function doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with non-function original', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withExactArgs('must provide one or more functions to wrap on modules')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.massWrap({ orange: 'slices' }, 'orange', () => {});
  }, 'wrapping non-function original doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});

test('wrap called with non-function wrapper', (t) => {
  t.plan(2);

  const mock = sinon.expectation
    .create('logger')
    .withArgs('must provide one or more functions to wrap on modules')
    .once();
  shimmer({ logger: mock });

  t.doesNotThrow(() => {
    shimmer.massWrap({ orange: function () {} }, 'orange', 'hamchunx');
  }, 'wrapping with non-function wrapper doesn\'t throw');

  t.doesNotThrow(() => {
    mock.verify();
  }, 'logger was called with the expected message');
});
