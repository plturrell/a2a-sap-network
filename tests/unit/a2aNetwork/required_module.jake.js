const { task, namespace } = require(`${process.env.PROJECT_DIR}/lib/jake`);

namespace('usingRequire', () => {
  task('test', () => {
    console.log('howdy test');
  });
});



