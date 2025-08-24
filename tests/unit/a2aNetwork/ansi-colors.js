const test = require('tape');
const table = require('../');
const color = require('cli-color');
const ansiTrim = require('cli-color/lib/trim');

test('center', (t) => {
    t.plan(1);
    const opts = {
        align: [ 'l', 'c', 'l' ],
        stringLength: function(s) { return ansiTrim(s).length; }
    };
    const s = table([
        [
            color.red('Red'), color.green('Green'), color.blue('Blue')
        ],
        [
            color.bold('Bold'), color.underline('Underline'),
            color.italic('Italic')
        ],
        [
            color.inverse('Inverse'), color.strike('Strike'),
            color.blink('Blink')
        ],
        [ 'bar', '45', 'lmno' ]
    ], opts);
    t.equal(ansiTrim(s), [
        'Red        Green    Blue',
        'Bold     Underline  Italic',
        'Inverse    Strike   Blink',
        'bar          45     lmno'
    ].join('\n'));
});
