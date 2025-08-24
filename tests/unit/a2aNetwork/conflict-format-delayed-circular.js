
const test = require('tap').test;
const defaultFormater = require('../../format.js');
const produce = require('../produce.js');

const chain = require('../../');

test('set Error.prepareStackTrace uses stack-chain formater', (t) => {
  // Save original formatter
  const restore = Error.prepareStackTrace;

  // Overwrite formatter
  Error.prepareStackTrace = function (error, frames) {
    if (error.test) {
      Object.defineProperty(error, '__some_secret', {
        value: 'you can\'t compare pain.'
      });
    }

    // Maintain .stack format
    return restore(error, frames);
  };

  // Prope the error using custom prepareStackTrace
  const testError = new Error();
  testError.test = true;
  testError.stack;
  t.equal(testError.__some_secret, 'you can\'t compare pain.');

  // Restore
  Error.prepareStackTrace = restore;

  t.equal(produce.real(3), produce.fake([
    'Error: trace',
    '    at {where}:18:17',
    '    at deepStack ({where}:5:5)',
    '    at deepStack ({where}:7:5)'
  ]));

  t.end();
});

test('set Error.prepareStackTrace uses other formater', (t) => {
  // Another module sets up a formater
  Error.prepareStackTrace = function () {
    return 'custom';
  };

  // Save original formatter
  const restore = Error.prepareStackTrace;

  // Overwrite formatter
  Error.prepareStackTrace = function (error, frames) {
    if (error.test) {
      Object.defineProperty(error, '__some_secret', {
        value: 'you can\'t compare pain.'
      });
    }

    // Maintain .stack format
    return restore(error, frames);
  };

  // Prope the error using custom prepareStackTrace
  const testError = new Error();
  testError.test = true;
  testError.stack;
  t.equal(testError.__some_secret, 'you can\'t compare pain.');

  // Restore
  Error.prepareStackTrace = restore;

  t.equal(produce.real(3), 'custom');

  // Perform an actual restore of the formater, to prevent test conflicts
  chain.format.restore();

  t.equal(produce.real(3), produce.fake([
    'Error: trace',
    '    at {where}:18:17',
    '    at deepStack ({where}:5:5)',
    '    at deepStack ({where}:7:5)'
  ]));

  t.end();
});
