const common = require('../common');
const assert = common.assert;
const retry = require(`${common.dir.lib  }/retry`);

(function testForeverUsesFirstTimeout() {
  const operation = retry.operation({
    retries: 0,
    minTimeout: 100,
    maxTimeout: 100,
    forever: true
  });

  operation.attempt((numAttempt) => {
    console.log('>numAttempt', numAttempt);
    const err = new Error('foo');
    if (numAttempt == 10) {
      operation.stop();
    }

    if (operation.retry(err)) {
      return;
    }
  });
})();
