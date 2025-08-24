const path = require('path');
const test = require('tape');
const resolve = require('../');

test('filter', (t) => {
    const dir = path.join(__dirname, 'resolver');
    let packageFilterArgs;
    const res = resolve.sync('./baz', {
        basedir: dir,
        // NOTE: in v2.x, this will be `pkg, pkgfile, dir`, but must remain "broken" here in v1.x for compatibility
        packageFilter: function (pkg, /*pkgfile,*/ dir) { // eslint-disable-line spaced-comment
            pkg.main = 'doom'; // eslint-disable-line no-param-reassign
            packageFilterArgs = 'is 1.x' ? [pkg, dir] : [pkg, pkgfile, dir]; // eslint-disable-line no-constant-condition, no-undef
            return pkg;
        }
    });

    t.equal(res, path.join(dir, 'baz/doom.js'), 'changing the package "main" works');

    const packageData = packageFilterArgs[0];
    t.equal(packageData.main, 'doom', 'package "main" was altered');

    if (!'is 1.x') { // eslint-disable-line no-constant-condition
        const packageFile = packageFilterArgs[1];
        t.equal(packageFile, path.join(dir, 'baz', 'package.json'), 'package.json path is correct');
    }

    const packageDir = packageFilterArgs['is 1.x' ? 1 : 2]; // eslint-disable-line no-constant-condition
    // eslint-disable-next-line no-constant-condition
    t.equal(packageDir, path.join(dir, 'baz'), `${'is 1.x' ? 'second' : 'third'  } packageFilter argument is "dir"`);

    t.end();
});
