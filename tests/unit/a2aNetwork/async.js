const from = require('from');
const through = require('../');

const tape = require('tape');

tape('simple async example', (t) => {
 
  let n = 0, expected = [1,2,3,4,5], actual = [];
  from(expected)
  .pipe(through(function(data) {
    this.pause();
    n ++;
    setTimeout(() =>{
      console.log('pushing data', data);
      this.push(data);
      this.resume();
    }, 300);
  })).pipe(through(function(data) {
    console.log('pushing data second time', data);
    this.push(data);
  })).on('data', (d) => {
    actual.push(d);
  }).on('end', () => {
    t.deepEqual(actual, expected);
    t.end();
  });

});
