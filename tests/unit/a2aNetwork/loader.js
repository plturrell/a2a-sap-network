const fs = require('fs');
const _ = require('underscore');

function load(filename) {
    const ret = [];
    const nums = fs.readFileSync(filename, 'ascii').split('\n');
    nums.forEach((s) => {
        if(s.length) {
            const n = s*1;
            ret.push(n);
        }
    });

    return ret;
}

function get_inserts(tests) {
    return _.select(tests, (n) => { return n > 0; });
}

function get_removes(tests) {
    return _.select(tests, (n) => { return n < 0; });
}

function new_tree(tree_type) {
    return new tree_type((a,b) => { return a - b; });
}

function build_tree(tree_type, inserts) {
    const tree = new_tree(tree_type);
    
    inserts.forEach((n) => {
        tree.insert(n);
    });

    return tree;
}

function load_tree(tree_type, filename) {
    const tests = load(filename);
    const inserts = get_inserts(tests);
    return build_tree(tree_type, inserts);
}

module.exports = {
    load: load,
    get_inserts: get_inserts,
    get_removes: get_removes,
    new_tree: new_tree,
    build_tree: build_tree,
    load_tree: load_tree
};
