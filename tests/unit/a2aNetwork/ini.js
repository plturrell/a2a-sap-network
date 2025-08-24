const cc =require('../lib/utils');
const INI = require('ini');
const assert = require('assert');

function test(obj) {

  let _json, _ini;
  const json = cc.parse (_json = JSON.stringify(obj));
  const ini = cc.parse (_ini = INI.stringify(obj));
  console.log(_ini, _json);
  assert.deepEqual(json, ini);
}


test({hello: true});

