const exit = require('../../lib/exit');

const errorCode = process.argv[2];
const max = process.argv[3];
const modes = process.argv.slice(4);

function stdout(message) {
  if (modes.indexOf('stdout') === -1) { return; }
  process.stdout.write(`stdout ${  message  }\n`);
}

function stderr(message) {
  if (modes.indexOf('stderr') === -1) { return; }
  process.stderr.write(`stderr ${  message  }\n`);
}

for (let i = 0; i < max; i++) {
  stdout(i);
  stderr(i);
}

exit(errorCode);

stdout('fail');
stderr('fail');
