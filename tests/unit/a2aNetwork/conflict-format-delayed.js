
const test = require('tap').test;
const defaultFormater = require('../../format.js');
const produce = require('../produce.js');

const chain = require('../../');

// Set a formater after stack-chain is required
function prepareStackTrace(error, frames) {
  if (error.test) {
    const lines = [];
        lines.push(error.toString());

    for (let i = 0, l = frames.length; i < l; i++) {
        lines.push(frames[i].getFunctionName());
    }

    return lines.join('\n');
  }

  return defaultFormater(error, frames);
}

test('set Error.prepareStackTrace after require', (t) => {
  t.test('set prepareStackTrace', (t) => {
    Error.prepareStackTrace = prepareStackTrace;
    t.end();
  });

  t.test('default formatter replaced', (t) => {
    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '',
      'deepStack',
      'deepStack'
    ]));

    t.end();
  });

  t.test('restore default formater', (t) => {
    chain.format.restore();

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.end();
});

test('set Error.prepareStackTrace after require to undefined', (t) => {
  t.test('set prepareStackTrace', (t) => {
    Error.prepareStackTrace = prepareStackTrace;
    t.end();
  });

  t.test('default formatter replaced', (t) => {
    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '',
      'deepStack',
      'deepStack'
    ]));

    t.end();
  });

  t.test('restore default formater', (t) => {
    Error.prepareStackTrace = undefined;

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.end();
});

test('set Error.prepareStackTrace after require to itself', (t) => {
  t.test('default formatter replaced', (t) => {
    const old = Error.prepareStackTrace;

    Error.prepareStackTrace = function () {
      return 'custom';
    };
    t.equal(new Error().stack, 'custom');

    Error.prepareStackTrace = old;

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.end();
});
