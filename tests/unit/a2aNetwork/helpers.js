const exec = require('child_process').exec;

var helpers = new (function () {
  let _tests;
  let _names = [];
  let _name;
  let _callback;
  const _runner = function () {
    if ((_name = _names.shift())) {
      console.log(`Running ${  _name}`);
      _tests[_name]();
    }
    else {
      _callback();
    }
  };

  this.exec = function () {
    const args = Array.prototype.slice.call(arguments);
    let arg;
    let cmd = args.shift();
    let opts = {};
    let callback;
    // Optional opts/callback or callback/opts
    while ((arg = args.shift())) {
      if (typeof arg === 'function') {
        callback = arg;
      }
      else {
        opts = arg;
      }
    }

    cmd += ' --trace';
    const execOpts = opts.execOpts ? opts.execOpts : {};
    exec(cmd, execOpts, (err, stdout, stderr) => {
      const out = helpers.trim(stdout);
      if (err) {
        if (opts.breakOnError === false) {
          return callback(err);
        }
        else {
          throw err;
        }
      }
      if (stderr) {
        callback(stderr);
      }
      else {
        callback(out);
      }
    });
  };

  this.trim = function (s) {
    const str = s || '';
    return str.replace(/^\s*|\s*$/g, '');
  };

  this.parse = function (s) {
    let str = s || '';
    str = helpers.trim(str);
    str = str.replace(/'/g, '"');
    return JSON.parse(str);
  };

  this.run = function (tests, callback) {
    _tests = tests;
    _names = Object.keys(tests);
    _callback = callback;
    _runner();
  };

  this.next = function () {
    _runner();
  };

})();

module.exports = helpers;
