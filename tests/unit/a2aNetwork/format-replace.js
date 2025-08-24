
const test = require('tap').test;
const chain = require('../../');
const defaultFormater = require('../../format.js');
const produce = require('../produce.js');

test('stack format part', (t) => {
  const format = function (error, frames) {
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

  t.test('no formatter set', (t) => {
    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('default formatter replaced', (t) => {
    chain.format.replace(format);

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '',
      'deepStack',
      'deepStack'
    ]));

    chain.format.restore();

    t.end();
  });

  t.test('restore default formater', (t) => {
    chain.format.replace(format);
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
