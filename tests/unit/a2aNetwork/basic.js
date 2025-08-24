const t = require('tap');
const fs = require('fs');
const path = require('path');
const fixture = path.resolve(__dirname, 'fixtures');
const meow = `${fixture  }/meow.cat`;
const mine = `${fixture  }/mine.cat`;
const ours = `${fixture  }/ours.cat`;
const fail = `${fixture  }/fail.false`;
const noent = `${fixture  }/enoent.exe`;
const mkdirp = require('mkdirp');
const rimraf = require('rimraf');

const isWindows = process.platform === 'win32';
const hasAccess = typeof fs.access === 'function';
const winSkip = isWindows && 'windows';
const accessSkip = !hasAccess && 'no fs.access function';
const hasPromise = typeof Promise === 'function';
const promiseSkip = !hasPromise && 'no global Promise';

function reset () {
  delete require.cache[require.resolve('../')];
  return require('../');
}

t.test('setup fixtures', (t) => {
  rimraf.sync(fixture);
  mkdirp.sync(fixture);
  fs.writeFileSync(meow, '#!/usr/bin/env cat\nmeow\n');
  fs.chmodSync(meow, parseInt('0755', 8));
  fs.writeFileSync(fail, '#!/usr/bin/env false\n');
  fs.chmodSync(fail, parseInt('0644', 8));
  fs.writeFileSync(mine, '#!/usr/bin/env cat\nmine\n');
  fs.chmodSync(mine, parseInt('0744', 8));
  fs.writeFileSync(ours, '#!/usr/bin/env cat\nours\n');
  fs.chmodSync(ours, parseInt('0754', 8));
  t.end();
});

t.test('promise', { skip: promiseSkip }, (t) => {
  const isexe = reset();
  t.test('meow async', (t) => {
    isexe(meow).then((is) => {
      t.ok(is);
      t.end();
    });
  });
  t.test('fail async', (t) => {
    isexe(fail).then((is) => {
      t.notOk(is);
      t.end();
    });
  });
  t.test('noent async', (t) => {
    isexe(noent).catch((er) => {
      t.ok(er);
      t.end();
    });
  });
  t.test('noent ignore async', (t) => {
    isexe(noent, { ignoreErrors: true }).then((is) => {
      t.notOk(is);
      t.end();
    });
  });
  t.end();
});

t.test('no promise', (t) => {
  global.Promise = null;
  const isexe = reset();
  t.throws('try to meow a promise', () => {
    isexe(meow);
  });
  t.end();
});

t.test('access', { skip: accessSkip || winSkip }, (t) => {
  runTest(t);
});

t.test('mode', { skip: winSkip }, (t) => {
  delete fs.access;
  delete fs.accessSync;
  const isexe = reset();
  t.ok(isexe.sync(ours, { uid: 0, gid: 0 }));
  t.ok(isexe.sync(mine, { uid: 0, gid: 0 }));
  runTest(t);
});

t.test('windows', (t) => {
  global.TESTING_WINDOWS = true;
  const pathExt = '.EXE;.CAT;.CMD;.COM';
  t.test('pathExt option', (t) => {
    runTest(t, { pathExt: '.EXE;.CAT;.CMD;.COM' });
  });
  t.test('pathExt env', (t) => {
    process.env.PATHEXT = pathExt;
    runTest(t);
  });
  t.test('no pathExt', (t) => {
    // with a pathExt of '', any filename is fine.
    // so the "fail" one would still pass.
    runTest(t, { pathExt: '', skipFail: true });
  });
  t.test('pathext with empty entry', (t) => {
    // with a pathExt of '', any filename is fine.
    // so the "fail" one would still pass.
    runTest(t, { pathExt: `;${  pathExt}`, skipFail: true });
  });
  t.end();
});

t.test('cleanup', (t) => {
  rimraf.sync(fixture);
  t.end();
});

function runTest (t, options) {
  const isexe = reset();

  const optionsIgnore = Object.create(options || {});
  optionsIgnore.ignoreErrors = true;

  if (!options || !options.skipFail) {
    t.notOk(isexe.sync(fail, options));
  }
  t.notOk(isexe.sync(noent, optionsIgnore));
  if (!options) {
    t.ok(isexe.sync(meow));
  } else {
    t.ok(isexe.sync(meow, options));
  }

  t.ok(isexe.sync(mine, options));
  t.ok(isexe.sync(ours, options));
  t.throws(() => {
    isexe.sync(noent, options);
  });

  t.test('meow async', (t) => {
    if (!options) {
      isexe(meow, (er, is) => {
        if (er) {
          throw er;
        }
        t.ok(is);
        t.end();
      });
    } else {
      isexe(meow, options, (er, is) => {
        if (er) {
          throw er;
        }
        t.ok(is);
        t.end();
      });
    }
  });

  t.test('mine async', (t) => {
    isexe(mine, options, (er, is) => {
      if (er) {
        throw er;
      }
      t.ok(is);
      t.end();
    });
  });

  t.test('ours async', (t) => {
    isexe(ours, options, (er, is) => {
      if (er) {
        throw er;
      }
      t.ok(is);
      t.end();
    });
  });

  if (!options || !options.skipFail) {
    t.test('fail async', (t) => {
      isexe(fail, options, (er, is) => {
        if (er) {
          throw er;
        }
        t.notOk(is);
        t.end();
      });
    });
  }

  t.test('noent async', (t) => {
    isexe(noent, options, (er, is) => {
      t.ok(er);
      t.notOk(is);
      t.end();
    });
  });

  t.test('noent ignore async', (t) => {
    isexe(noent, optionsIgnore, (er, is) => {
      if (er) {
        throw er;
      }
      t.notOk(is);
      t.end();
    });
  });

  t.test('directory is not executable', (t) => {
    isexe(__dirname, options, (er, is) => {
      if (er) {
        throw er;
      }
      t.notOk(is);
      t.end();
    });
  });

  t.end();
}
