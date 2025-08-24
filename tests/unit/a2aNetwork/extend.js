
const test = require('tap').test;
const chain = require('../../');
const produce = require('../produce.js');

test('stack extend part', (t) => {
  const modify = function (text) {
    return function (error, frames) {
      if (error.test) {
        frames.splice(1, 0, text);
      }

      return frames;
    };
  };

  t.test('no extend modifier attached', (t) => {
    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.end();
  });

  t.test('attach modifier', (t) => {
    const wonderLand = modify('wonder land');

    chain.extend.attach(wonderLand);

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at wonder land',
      '    at deepStack ({where}:5:5)'
    ]));

    chain.extend.deattach(wonderLand);

    t.end();
  });

  t.test('deattach modifier', (t) => {
    const wonderLand = modify('wonder land');

    chain.extend.attach(wonderLand);
    t.equal(chain.extend.deattach(wonderLand), true);

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at deepStack ({where}:5:5)',
      '    at deepStack ({where}:7:5)'
    ]));

    t.equal(chain.extend.deattach(wonderLand), false);

    t.end();
  });

  t.test('execution order', (t) => {
    const wonderLand = modify('wonder land');
    const outerSpace = modify('outer space');

    chain.extend.attach(wonderLand);
    chain.extend.attach(outerSpace);

    t.equal(produce.real(3), produce.fake([
      'Error: trace',
      '    at {where}:18:17',
      '    at outer space',
      '    at wonder land'
    ]));

    chain.extend.deattach(wonderLand);
    chain.extend.deattach(outerSpace);

    t.end();
  });

  t.end();
});
