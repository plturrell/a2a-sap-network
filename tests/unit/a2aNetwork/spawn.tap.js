const test   = require('tap').test
  , assert = require('assert')
  ;

if (!global.setImmediate) global.setImmediate = setTimeout;

if (!process.addAsyncListener) require('../index.js');

const childProcess = require('child_process')
  , exec         = childProcess.exec
  , execFile     = childProcess.execFile
  , spawn        = childProcess.spawn
  ;

test('ChildProcess', (t) => {
  t.plan(3);

  t.test('exec', (t) => {
    t.plan(3);

    let active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(() => {
      t.equal(active, 1,
        'after tick: 1st context');
      const child = exec('node --version');
      child.on('exit', (code) => {
        t.ok(active >= 2,
          'after exec#exit: entered additional contexts');
      });
    });
  });

  t.test('execFile', (t) => {
    t.plan(3);

    let active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(() => {
      t.equal(active, 1,
        'after nextTick: 1st context');
      execFile('node', ['--version'], (err, code) => {
        t.ok(active >= 2,
          'after execFile: entered additional contexts');
      });
    });
  });

  t.test('spawn', (t) => {
    t.plan(3);

    let active
      , cntr   = 0
      ;

    process.addAsyncListener(
      {
        create : function () { return { val : ++cntr }; },
        before : function (context, data) { active = data.val; },
        after  : function () { active = null; }
      }
    );

    t.equal(active, undefined,
      'starts in initial context');
    process.nextTick(() => {
      t.equal(active, 1,
        'after tick: 1st context');
      const child = spawn('node', ['--version']);
      child.on('exit', (code) => {
        t.ok(active >= 2,
          'after spawn#exit: entered additional contexts');
      });
    });
  });
});
