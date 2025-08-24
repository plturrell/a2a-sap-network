const path = require('path');
const test = require('tape');
const resolve = require('../');

test('filter', (t) => {
    t.plan(4);
    const dir = path.join(__dirname, 'resolver');
    let packageFilterArgs;
    resolve('./baz', {
        basedir: dir,
        packageFilter: function (pkg, pkgfile) {
            pkg.main = 'doom'; // eslint-disable-line no-param-reassign
            packageFilterArgs = [pkg, pkgfile];
            return pkg;
        }
    }, (err, res, pkg) => {
        if (err) t.fail(err);

        t.equal(res, path.join(dir, 'baz/doom.js'), 'changing the package "main" works');

        const packageData = packageFilterArgs[0];
        t.equal(pkg, packageData, 'first packageFilter argument is "pkg"');
        t.equal(packageData.main, 'doom', 'package "main" was altered');

        const packageFile = packageFilterArgs[1];
        t.equal(
            packageFile,
            path.join(dir, 'baz/package.json'),
            'second packageFilter argument is "pkgfile"'
        );

        t.end();
    });
});
