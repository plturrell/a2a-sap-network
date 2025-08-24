
const test = require('tap').test;
const defaultFormater = require('../../format.js');
const produce = require('../produce.js');

// Set a formater before stack-chain is required
Error.prepareStackTrace = function (error, frames) {
  if (error.test) {
    const lines = [];
        lines.push(error.toString());

    for (let i = 0, l = frames.length; i < l; i++) {
        lines.push(frames[i].getFunctionName());
    }

    return lines.join('\n');
  }

  return defaultFormater(error, frames);
};

const chain = require('../../');

test('set Error.prepareStackTrace before require', (t) => {
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
