'use strict';

if (process.env.hasOwnProperty('ASYNC_HOOK_TEST_CHILD')) {
  const asyncHook = require('../');

  asyncHook.enable();

  setImmediate(() => {
    throw new Error('test error');
  });
} else {
  const spawn = require('child_process').spawn;
  const endpoint = require('endpoint');

  const child = spawn(process.execPath, [__filename], {
    env: Object.assign({ ASYNC_HOOK_TEST_CHILD: '' }, process.env),
    stdio: ['ignore', 1, 'pipe']
  });

  let stderr = null;
  child.stderr.pipe(endpoint((err, _stderr) => {
    if (err) throw err;
    stderr = _stderr;
  }));

  child.once('close', (statusCode) => {
    if (statusCode !== 1 || stderr.toString().indexOf('test error') === -1) {
      process.stderr.write(stderr);
      process.exit(statusCode);
    }
  });
}
