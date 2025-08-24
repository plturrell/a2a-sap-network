'use strict';

// I love when a tap.plan() comes together
console.log('1..1');

process.on('uncaughtException', (err) => {
  if (err.message === 'oops') {
    console.log('ok got expected message: %s', err.message);
  }
  else {
    throw err;
  }
});

const cls = require('../context.js');
const ns = cls.createNamespace('x');
ns.run(() => { throw new Error('oops'); });
