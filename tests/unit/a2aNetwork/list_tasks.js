const PROJECT_DIR = process.env.PROJECT_DIR;
const JAKE_CMD = `${PROJECT_DIR}/bin/cli.js`;

const assert = require('assert');
const proc = require('child_process');

suite('listTasks', () => {
  test('execute "jake -T" without any errors', () => {
    const message = 'cannot run "jake -T" command';
    const listTasks = function () {
      proc.execFileSync(JAKE_CMD, ['-T']);
    };
    assert.doesNotThrow(listTasks, TypeError, message);
  });
});
