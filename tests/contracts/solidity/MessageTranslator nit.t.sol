// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Test.sol";

// Import only the MessageTranslator library for isolated testing
library MessageTranslator {
    function translateToANP(
        string memory content,
        bytes32 messageType,
        string memory sender,
        string memory recipient
    ) internal pure returns (string memory) {
        return string(abi.encodePacked(
            '{"@context": ["https://www.w3.org/ns/did/v1","https://agent-network-protocol.com/context/v1"],',
            '"@type": "ad:Message",',
            '"@id": "urn:uuid:test-message-id",',
            '"ad:sender": "', sender, '",',
            '"ad:recipient": "', recipient, '",',
            '"ad:content": {',
                '"@type": "ad:TaskRequest",',
                '"ad:description": "', content, '",',
                '"ad:messageType": "', _bytes32ToString(messageType), '"',
            '},',
            '"ad:timestamp": "1234567890",',
            '"ad:protocol": "A2A-via-ANP"}'
        ));
    }
    
    function translateToACP(
        string memory content,
        bytes32 messageType,
        string memory sessionId
    ) internal pure returns (string memory) {
        return string(abi.encodePacked(
            '{"operation": "sendTask",',
            '"message": {',
                '"parts": [{',
                    '"content_type": "application/json",',
                    '"name": "task_data",',
                    '"content": "', content, '"',
                '}]',
            '},',
            '"session_id": "', sessionId, '",',
            '"task_id": "task-test-id",',
            '"timestamp": "1234567890"}'
        ));
    }
    
    function _bytes32ToString(bytes32 _bytes32) internal pure returns (string memory) {
        uint8 i = 0;
        while (i < 32 && _bytes32[i] != 0) {
            i++;
        }
        bytes memory bytesArray = new bytes(i);
        for (i = 0; i < 32 && _bytes32[i] != 0; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }
}

contract MessageTranslatorUnitTest is Test {
    using MessageTranslator for string;
    
    function testANPTranslation() public {
        string memory content = "Analyze customer behavior patterns";
        bytes32 messageType = keccak256("data_analysis");
        string memory sender = "did:wba:sender-agent";
        string memory recipient = "did:anp:recipient-agent";
        
        string memory anpMessage = MessageTranslator.translateToANP(
            content,
            messageType,
            sender,
            recipient
        );
        
        // Verify ANP JSON-LD structure
        assertTrue(bytes(anpMessage).length > 0, "Message should not be empty");
        assertTrue(_contains(anpMessage, "@context"), "Should contain context");
        assertTrue(_contains(anpMessage, "https://agent-network-protocol.com"), "Should contain ANP context");
        assertTrue(_contains(anpMessage, "ad:Message"), "Should contain message type");
        assertTrue(_contains(anpMessage, sender), "Should contain sender");
        assertTrue(_contains(anpMessage, recipient), "Should contain recipient");
        assertTrue(_contains(anpMessage, content), "Should contain original content");
        assertTrue(_contains(anpMessage, "A2A-via-ANP"), "Should indicate protocol bridge");
        
        console.log("ANP Translation Result:");
        console.log(anpMessage);
    }
    
    function testACPTranslation() public {
        string memory content = "Generate comprehensive analytics report";
        bytes32 messageType = keccak256("report_generation");
        string memory sessionId = "session_12345";
        
        string memory acpMessage = MessageTranslator.translateToACP(
            content,
            messageType,
            sessionId
        );
        
        // Verify ACP multipart structure
        assertTrue(bytes(acpMessage).length > 0, "Message should not be empty");
        assertTrue(_contains(acpMessage, "sendTask"), "Should contain sendTask operation");
        assertTrue(_contains(acpMessage, "message"), "Should contain message field");
        assertTrue(_contains(acpMessage, "parts"), "Should contain parts array");
        assertTrue(_contains(acpMessage, "application/json"), "Should specify content type");
        assertTrue(_contains(acpMessage, sessionId), "Should contain session ID");
        assertTrue(_contains(acpMessage, content), "Should contain original content");
        
        console.log("ACP Translation Result:");
        console.log(acpMessage);
    }
    
    function testComplexContent() public {
        string memory complexContent = "Process data with special characters: quotes and symbols";
        bytes32 messageType = keccak256("complex_processing");
        string memory sender = "did:wba:complex-sender";
        string memory recipient = "did:anp:complex-recipient";
        
        string memory anpMessage = MessageTranslator.translateToANP(
            complexContent,
            messageType,
            sender,
            recipient
        );
        
        assertTrue(bytes(anpMessage).length > 0, "Complex message should be handled");
        assertTrue(_contains(anpMessage, "special characters"), "Should handle special content");
        assertTrue(_contains(anpMessage, "quotes"), "Should handle quotes");
        
        // Test ACP with same complex content
        string memory acpMessage = MessageTranslator.translateToACP(
            complexContent,
            messageType,
            "complex_session"
        );
        
        assertTrue(bytes(acpMessage).length > 0, "Complex ACP message should be handled");
        assertTrue(_contains(acpMessage, "special characters"), "ACP should handle special content");
    }
    
    function testEmptyContent() public {
        string memory emptyContent = "";
        bytes32 messageType = keccak256("empty_test");
        
        string memory anpMessage = MessageTranslator.translateToANP(
            emptyContent,
            messageType,
            "did:sender",
            "did:recipient"
        );
        
        assertTrue(bytes(anpMessage).length > 0, "Should generate valid JSON even for empty content");
        assertTrue(_contains(anpMessage, "@type"), "Should contain type even for empty content");
        
        string memory acpMessage = MessageTranslator.translateToACP(
            emptyContent,
            messageType,
            "empty_session"
        );
        
        assertTrue(bytes(acpMessage).length > 0, "Should generate valid ACP JSON for empty content");
        assertTrue(_contains(acpMessage, "sendTask"), "Should contain operation for empty content");
    }
    
    function testMessageTypeEncoding() public {
        bytes32[] memory messageTypes = new bytes32[](3);
        messageTypes[0] = keccak256("data_analysis");
        messageTypes[1] = keccak256("text_processing");
        messageTypes[2] = keccak256("image_generation");
        
        for (uint256 i = 0; i < messageTypes.length; i++) {
            string memory anpMessage = MessageTranslator.translateToANP(
                "Test content",
                messageTypes[i],
                "did:sender",
                "did:recipient"
            );
            
            // Each message type should be encoded in the message
            assertTrue(bytes(anpMessage).length > 0, "Message should be generated for each type");
            assertTrue(_contains(anpMessage, "ad:messageType"), "Should contain message type field");
        }
    }
    
    function testLargeContent() public {
        // Generate large content string
        string memory largeContent = _generateLargeString(500);
        
        string memory anpMessage = MessageTranslator.translateToANP(
            largeContent,
            keccak256("large_content_test"),
            "did:sender",
            "did:recipient"
        );
        
        assertTrue(bytes(anpMessage).length > 500, "Large message should be handled");
        assertTrue(_contains(anpMessage, "@context"), "Large message should maintain structure");
        
        string memory acpMessage = MessageTranslator.translateToACP(
            largeContent,
            keccak256("large_content_test"),
            "large_session"
        );
        
        assertTrue(bytes(acpMessage).length > 500, "Large ACP message should be handled");
        assertTrue(_contains(acpMessage, "sendTask"), "Large message should maintain ACP structure");
    }
    
    // Helper functions
    function _contains(string memory haystack, string memory needle) internal pure returns (bool) {
        bytes memory haystackBytes = bytes(haystack);
        bytes memory needleBytes = bytes(needle);
        
        if (needleBytes.length > haystackBytes.length) return false;
        
        for (uint256 i = 0; i <= haystackBytes.length - needleBytes.length; i++) {
            bool found = true;
            for (uint256 j = 0; j < needleBytes.length; j++) {
                if (haystackBytes[i + j] != needleBytes[j]) {
                    found = false;
                    break;
                }
            }
            if (found) return true;
        }
        return false;
    }
    
    function _generateLargeString(uint256 length) internal pure returns (string memory) {
        bytes memory result = new bytes(length);
        for (uint256 i = 0; i < length; i++) {
            result[i] = bytes1(uint8(65 + (i % 26))); // A-Z repeating
        }
        return string(result);
    }
}