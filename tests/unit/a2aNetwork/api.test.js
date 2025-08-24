/*global fetch*/
'use strict';

require('../fetch-npm-node');
const expect = require('chai').expect;
const nock = require('nock');
const good = 'hello world. 你好世界。';
const bad = 'good bye cruel world. 再见残酷的世界。';

function responseToText(response) {
	if (response.status >= 400) throw new Error('Bad server response');
	return response.text();
}

describe('fetch', () => {

	before(() => {
		nock('https://mattandre.ws')
			.get('/succeed.txt')
			.reply(200, good);
		nock('https://mattandre.ws')
			.get('/fail.txt')
			.reply(404, bad);
	});

	it('should be defined', () => {
		expect(fetch).to.be.a('function');
	});

	it('should facilitate the making of requests', (done) => {
		fetch('//mattandre.ws/succeed.txt')
			.then(responseToText)
			.then((data) => {
				expect(data).to.equal(good);
				done();
			})
			.catch(done);
	});

	it('should do the right thing with bad requests', (done) => {
		fetch('//mattandre.ws/fail.txt')
			.then(responseToText)
			.catch((err) => {
				expect(err.toString()).to.equal('Error: Bad server response');
				done();
			})
			.catch(done);
	});

});
