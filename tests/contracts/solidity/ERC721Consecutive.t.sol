// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

// solhint-disable func-name-mixedcase

import "../../../../contracts/token/ERC721/extensions/ERC721Consecutive.sol";
import "forge-std/Test.sol";

function toSingleton(address account) pure returns (address[] memory) {
    address[] memory accounts = new address[](1);
    accounts[0] = account;
    return accounts;
}

contract ERC721ConsecutiveTarget is StdUtils, ERC721Consecutive {
    uint256 public totalMinted = 0;

    constructor(address[] memory receivers, uint256[] memory batches) ERC721("", "") {
        uint256 batchLength = batches.length;
        uint256 receiverLength = receivers.length;
        for (uint256 i; i < batchLength; ++i) {
            address receiver = receivers[i % receiverLength];
            uint96 batchSize = uint96(bound(batches[i], 0, _maxBatchSize()));
            _mintConsecutive(receiver, batchSize);
            totalMinted += batchSize;
        }
    }

    /**
     * @notice Burns a specific token
     * @param tokenId The ID of the token to burn
     */
    function burn(uint256 tokenId) external {
        _burn(tokenId);
    }
}

contract ERC721ConsecutiveTest is Test {
    /**
     * @notice Tests that the balance matches the total minted tokens
     * @param receiver The address receiving tokens
     * @param batches Array of batch sizes to mint
     */
    function test_balance(address receiver, uint256[] calldata batches) public {
        vm.assume(receiver != address(0));

        ERC721ConsecutiveTarget token = new ERC721ConsecutiveTarget(toSingleton(receiver), batches);

        assertEq(token.balanceOf(receiver), token.totalMinted());
    }

    /**
     * @notice Tests token ownership and invalid token ID handling
     * @param receiver The address receiving tokens
     * @param batches Array of batch sizes to mint
     * @param unboundedTokenId Two token IDs for testing valid and invalid cases
     */
    function test_ownership(address receiver, uint256[] calldata batches, uint256[2] calldata unboundedTokenId) public {
        vm.assume(receiver != address(0));

        ERC721ConsecutiveTarget token = new ERC721ConsecutiveTarget(toSingleton(receiver), batches);

        if (token.totalMinted() > 0) {
            uint256 validTokenId = bound(unboundedTokenId[0], 0, token.totalMinted() - 1);
            assertEq(token.ownerOf(validTokenId), receiver);
        }

        uint256 invalidTokenId = bound(unboundedTokenId[1], token.totalMinted(), type(uint256).max);
        vm.expectRevert();
        token.ownerOf(invalidTokenId);
    }

    function test_burn(address receiver, uint256[] calldata batches, uint256 unboundedTokenId) public {
        vm.assume(receiver != address(0));

        ERC721ConsecutiveTarget token = new ERC721ConsecutiveTarget(toSingleton(receiver), batches);

        // only test if we minted at least one token
        uint256 supply = token.totalMinted();
        vm.assume(supply > 0);

        // burn a token in [0; supply[
        uint256 tokenId = bound(unboundedTokenId, 0, supply - 1);
        token.burn(tokenId);

        // balance should have decreased
        assertEq(token.balanceOf(receiver), supply - 1);

        // token should be burnt
        vm.expectRevert();
        token.ownerOf(tokenId);
    }

    function test_transfer(
        address[2] calldata accounts,
        uint256[2] calldata unboundedBatches,
        uint256[2] calldata unboundedTokenId
    ) public {
        vm.assume(accounts[0] != address(0));
        vm.assume(accounts[1] != address(0));
        vm.assume(accounts[0] != accounts[1]);

        address[] memory receivers = new address[](2);
        receivers[0] = accounts[0];
        receivers[1] = accounts[1];

        // We assume _maxBatchSize is 5000 (the default). This test will break otherwise.
        uint256[] memory batches = new uint256[](2);
        batches[0] = bound(unboundedBatches[0], 1, 5000);
        batches[1] = bound(unboundedBatches[1], 1, 5000);

        ERC721ConsecutiveTarget token = new ERC721ConsecutiveTarget(receivers, batches);

        uint256 tokenId0 = bound(unboundedTokenId[0], 0, batches[0] - 1);
        uint256 tokenId1 = bound(unboundedTokenId[1], 0, batches[1] - 1) + batches[0];

        assertEq(token.ownerOf(tokenId0), accounts[0]);
        assertEq(token.ownerOf(tokenId1), accounts[1]);
        assertEq(token.balanceOf(accounts[0]), batches[0]);
        assertEq(token.balanceOf(accounts[1]), batches[1]);

        vm.prank(accounts[0]);
        token.transferFrom(accounts[0], accounts[1], tokenId0);

        assertEq(token.ownerOf(tokenId0), accounts[1]);
        assertEq(token.ownerOf(tokenId1), accounts[1]);
        assertEq(token.balanceOf(accounts[0]), batches[0] - 1);
        assertEq(token.balanceOf(accounts[1]), batches[1] + 1);

        vm.prank(accounts[1]);
        token.transferFrom(accounts[1], accounts[0], tokenId1);

        assertEq(token.ownerOf(tokenId0), accounts[1]);
        assertEq(token.ownerOf(tokenId1), accounts[0]);
        assertEq(token.balanceOf(accounts[0]), batches[0]);
        assertEq(token.balanceOf(accounts[1]), batches[1]);
    }
}
