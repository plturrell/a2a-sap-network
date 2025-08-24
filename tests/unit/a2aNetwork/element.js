const inspect = require('../');
const test = require('tape');

test('element', (t) => {
    t.plan(3);
    const elem = {
        nodeName: 'div',
        attributes: [{ name: 'class', value: 'row' }],
        getAttribute: function (key) { return key; },
        childNodes: []
    };
    const obj = [1, elem, 3];
    t.deepEqual(inspect(obj), '[ 1, <div class="row"></div>, 3 ]');
    t.deepEqual(inspect(obj, { quoteStyle: 'single' }), '[ 1, <div class=\'row\'></div>, 3 ]');
    t.deepEqual(inspect(obj, { quoteStyle: 'double' }), '[ 1, <div class="row"></div>, 3 ]');
});

test('element no attr', (t) => {
    t.plan(1);
    const elem = {
        nodeName: 'div',
        getAttribute: function (key) { return key; },
        childNodes: []
    };
    const obj = [1, elem, 3];
    t.deepEqual(inspect(obj), '[ 1, <div></div>, 3 ]');
});

test('element with contents', (t) => {
    t.plan(1);
    const elem = {
        nodeName: 'div',
        getAttribute: function (key) { return key; },
        childNodes: [{ nodeName: 'b' }]
    };
    const obj = [1, elem, 3];
    t.deepEqual(inspect(obj), '[ 1, <div>...</div>, 3 ]');
});

test('element instance', (t) => {
    t.plan(1);
    const h = global.HTMLElement;
    global.HTMLElement = function (name, attr) {
        this.nodeName = name;
        this.attributes = attr;
    };
    global.HTMLElement.prototype.getAttribute = function () {};

    const elem = new global.HTMLElement('div', []);
    const obj = [1, elem, 3];
    t.deepEqual(inspect(obj), '[ 1, <div></div>, 3 ]');
    global.HTMLElement = h;
});
