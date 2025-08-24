require('@openzeppelin/test-helpers');
const { shouldBehaveLikeEscrow } = require('./Escrow.behavior');

const Escrow = artifacts.require('Escrow');

contract('Escrow', (accounts) => {
  const [owner, ...otherAccounts] = accounts;

  beforeEach(async function () {
    this.escrow = await Escrow.new({ from: owner });
  });

  shouldBehaveLikeEscrow(owner, otherAccounts);
});
