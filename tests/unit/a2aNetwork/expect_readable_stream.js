const TypedError = require('error/typed');

const MoreDataThanExpectedError = TypedError({
    type: 'more-data-than-expected',
    message: 'got more data than expected',
    numGot: null,
    numExpected: null,
    data: null
});

const LessDataThanExpectedError = TypedError({
    type: 'less-data-than-expected',
    message: 'got less data than expected',
    numGot: null,
    numExpected: null
});

function expectReadableStream(stream, expectedData, done) {
    let finished = false;
    let expectedI = 0;
    let numGot = 0;
    const unexpected = [];
    stream.on('data', onData);
    stream.on('error', finish);
    stream.on('end', finish);
    function onData(data) {
        numGot++;
        if (expectedI >= expectedData.length) {
            unexpected.push(data);
        } else {
            const expected = expectedData[expectedI++];
            expected(data);
        }
    }
    function finish(err) {
        if (finished) return;
        finished = true;
        stream.removeListener('error', finish);
        stream.removeListener('end', finish);
        if (err) {
            done(err);
        } else if (unexpected.length) {
            done(MoreDataThanExpectedError({
                numGot: numGot,
                numExpected: expectedData.length,
                data: unexpected
            }));
        } else if (expectedI < expectedData.length-1) {
            done(LessDataThanExpectedError({
                numGot: numGot,
                numExpected: expectedData.length
            }));
        } else {
            done(null);
        }
    }
}

module.exports = expectReadableStream;
