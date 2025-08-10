// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title MessageTranslator
 * @dev Advanced message format translator for converting between A2A, ANP, and ACP protocols.
 * Implements real JSON-LD parsing/generation and multipart message handling.
 */
library MessageTranslator {
    enum ProtocolType { A2A, ANP, ACP }
    
    struct MessagePart {
        string contentType;
        string name;
        bytes content;
        string contentUrl;
    }
    
    struct ANPMessage {
        string context;
        string messageType;
        string content;
        string sender;
        string recipient;
        uint256 timestamp;
        string id;
    }
    
    struct ACPMessage {
        string operation;
        MessagePart[] parts;
        string sessionId;
        string taskId;
    }
    
    // JSON-LD context templates
    string constant ANP_CONTEXT = '["https://www.w3.org/ns/did/v1","https://agent-network-protocol.com/context/v1"]';
    string constant ANP_MESSAGE_TYPE = "ad:Message";
    
    /**
     * @notice Convert A2A message to ANP JSON-LD format
     * @param content A2A message content
     * @param messageType A2A message type
     * @param sender Sender DID
     * @param recipient Recipient DID
     * @return ANP-formatted JSON-LD message
     */
    function translateToANP(
        string memory content,
        bytes32 messageType,
        string memory sender,
        string memory recipient
    ) internal view returns (string memory) {
        return string(abi.encodePacked(
            '{"@context": ', ANP_CONTEXT, ',',
            '"@type": "', ANP_MESSAGE_TYPE, '",',
            '"@id": "urn:uuid:', _generateMessageId(content, messageType), '",',
            '"ad:sender": "', sender, '",',
            '"ad:recipient": "', recipient, '",',
            '"ad:content": {',
                '"@type": "ad:TaskRequest",',
                '"ad:description": "', _escapeJson(content), '",',
                '"ad:messageType": "', _bytes32ToString(messageType), '"',
            '},',
            '"ad:timestamp": "', _getCurrentTimestamp(), '",',
            '"ad:protocol": "A2A-via-ANP"}'
        ));
    }
    
    /**
     * @notice Convert A2A message to ACP multipart format
     * @param content A2A message content
     * @param messageType A2A message type
     * @param sessionId Session identifier
     * @return ACP-formatted multipart message
     */
    function translateToACP(
        string memory content,
        bytes32 messageType,
        string memory sessionId
    ) internal view returns (string memory) {
        return string(abi.encodePacked(
            '{"operation": "sendTask",',
            '"message": {',
                '"parts": [{',
                    '"content_type": "application/json",',
                    '"name": "task_data",',
                    '"content": {',
                        '"description": "', _escapeJson(content), '",',
                        '"type": "', _bytes32ToString(messageType), '",',
                        '"source_protocol": "A2A"',
                    '}',
                '}]',
            '},',
            '"session_id": "', sessionId, '",',
            '"task_id": "', _generateTaskId(content, messageType), '",',
            '"timestamp": "', _getCurrentTimestamp(), '",',
            '"metadata": {',
                '"bridge_version": "1.0.0",',
                '"original_protocol": "A2A"',
            '}}'
        ));
    }
    
    /**
     * @notice Parse ANP JSON-LD message to A2A format
     * @param anpMessage ANP JSON-LD formatted message
     * @return content Extracted content
     * @return messageType Extracted message type
     * @return sender Extracted sender
     */
    function parseFromANP(string memory anpMessage) 
        internal 
        pure 
        returns (string memory content, bytes32 messageType, string memory sender) 
    {
        // Real JSON parsing implementation
        bytes memory messageBytes = bytes(anpMessage);
        
        // Extract content from "ad:content" -> "ad:description"
        content = _extractJsonValue(messageBytes, "ad:description");
        
        // Extract message type
        string memory typeStr = _extractJsonValue(messageBytes, "ad:messageType");
        messageType = _stringToBytes32(typeStr);
        
        // Extract sender
        sender = _extractJsonValue(messageBytes, "ad:sender");
        
        return (content, messageType, sender);
    }
    
    /**
     * @notice Parse ACP multipart message to A2A format
     * @param acpMessage ACP multipart formatted message
     * @return content Extracted content
     * @return messageType Extracted message type
     * @return sessionId Extracted session ID
     */
    function parseFromACP(string memory acpMessage)
        internal
        pure
        returns (string memory content, bytes32 messageType, string memory sessionId)
    {
        bytes memory messageBytes = bytes(acpMessage);
        
        // Extract from first part's content
        string memory partContent = _extractJsonValue(messageBytes, "content");
        content = _extractJsonValue(bytes(partContent), "description");
        
        // Extract message type from part content
        string memory typeStr = _extractJsonValue(bytes(partContent), "type");
        messageType = _stringToBytes32(typeStr);
        
        // Extract session ID
        sessionId = _extractJsonValue(messageBytes, "session_id");
        
        return (content, messageType, sessionId);
    }
    
    /**
     * @notice Convert ANP Agent Description to A2A Agent Card format
     * @param anpDescription ANP agent description JSON-LD
     * @return name Agent name
     * @return endpoint Agent endpoint
     * @return capabilities Agent capabilities
     */
    function parseANPAgentDescription(string memory anpDescription)
        internal
        pure
        returns (string memory name, string memory endpoint, bytes32[] memory capabilities)
    {
        bytes memory descBytes = bytes(anpDescription);
        
        name = _extractJsonValue(descBytes, "name");
        
        // Extract first interface as endpoint
        string memory interfacesJson = _extractJsonValue(descBytes, "ad:interfaces");
        endpoint = _extractArrayFirstElement(bytes(interfacesJson));
        
        // Extract capabilities
        string memory capabilitiesJson = _extractJsonValue(descBytes, "ad:capabilities");
        capabilities = _parseCapabilities(capabilitiesJson);
        
        return (name, endpoint, capabilities);
    }
    
    /**
     * @notice Convert ACP Agent Detail to A2A Agent Card format
     * @param acpDetail ACP agent detail JSON
     * @return name Agent name
     * @return endpoint Agent endpoint  
     * @return capabilities Agent capabilities
     */
    function parseACPAgentDetail(string memory acpDetail)
        internal
        pure
        returns (string memory name, string memory endpoint, bytes32[] memory capabilities)
    {
        bytes memory detailBytes = bytes(acpDetail);
        
        name = _extractJsonValue(detailBytes, "name");
        
        // Extract first endpoint
        string memory endpointsJson = _extractJsonValue(detailBytes, "endpoints");
        endpoint = _extractArrayFirstElement(bytes(endpointsJson));
        
        // Extract operations as capabilities
        string memory operationsJson = _extractJsonValue(detailBytes, "operations");
        capabilities = _parseOperationsAsCapabilities(operationsJson);
        
        return (name, endpoint, capabilities);
    }
    
    // Internal JSON parsing functions
    
    /**
     * @dev Extract JSON value by key (simplified but functional JSON parser)
     */
    function _extractJsonValue(bytes memory json, string memory key) 
        internal 
        pure 
        returns (string memory) 
    {
        bytes memory keyBytes = bytes(key);
        bytes memory searchPattern = abi.encodePacked('"', keyBytes, '":');
        
        // Find the key in the JSON
        int256 keyIndex = _indexOf(json, searchPattern);
        if (keyIndex < 0) return "";
        
        uint256 startIndex = uint256(keyIndex) + searchPattern.length;
        
        // Skip whitespace and quote
        while (startIndex < json.length && (json[startIndex] == ' ' || json[startIndex] == '\t')) {
            startIndex++;
        }
        
        if (startIndex >= json.length) return "";
        
        // Handle different value types
        if (json[startIndex] == '"') {
            // String value
            startIndex++; // Skip opening quote
            uint256 endIndex = startIndex;
            
            while (endIndex < json.length && json[endIndex] != '"') {
                if (json[endIndex] == '\\') endIndex++; // Skip escaped characters
                endIndex++;
            }
            
            if (endIndex > json.length) return "";
            
            bytes memory result = new bytes(endIndex - startIndex);
            for (uint256 i = 0; i < endIndex - startIndex; i++) {
                result[i] = json[startIndex + i];
            }
            return string(result);
            
        } else if (json[startIndex] == '{' || json[startIndex] == '[') {
            // Object or array value
            uint256 braceCount = 1;
            uint256 endIndex = startIndex + 1;
            bytes1 openBrace = json[startIndex];
            bytes1 closeBrace = (openBrace == '{') ? bytes1('}') : bytes1(']');
            
            while (endIndex < json.length && braceCount > 0) {
                if (json[endIndex] == openBrace) braceCount++;
                else if (json[endIndex] == closeBrace) braceCount--;
                endIndex++;
            }
            
            bytes memory result = new bytes(endIndex - startIndex);
            for (uint256 i = 0; i < endIndex - startIndex; i++) {
                result[i] = json[startIndex + i];
            }
            return string(result);
            
        } else {
            // Primitive value (number, boolean, null)
            uint256 endIndex = startIndex;
            while (endIndex < json.length && 
                   json[endIndex] != ',' && 
                   json[endIndex] != '}' && 
                   json[endIndex] != ']' &&
                   json[endIndex] != '\n' &&
                   json[endIndex] != '\r') {
                endIndex++;
            }
            
            bytes memory result = new bytes(endIndex - startIndex);
            for (uint256 i = 0; i < endIndex - startIndex; i++) {
                result[i] = json[startIndex + i];
            }
            return string(result);
        }
    }
    
    /**
     * @dev Find index of pattern in bytes array
     */
    function _indexOf(bytes memory data, bytes memory pattern) 
        internal 
        pure 
        returns (int256) 
    {
        if (pattern.length > data.length) return -1;
        
        for (uint256 i = 0; i <= data.length - pattern.length; i++) {
            bool found = true;
            for (uint256 j = 0; j < pattern.length; j++) {
                if (data[i + j] != pattern[j]) {
                    found = false;
                    break;
                }
            }
            if (found) return int256(i);
        }
        return -1;
    }
    
    /**
     * @dev Extract first element from JSON array
     */
    function _extractArrayFirstElement(bytes memory arrayJson) 
        internal 
        pure 
        returns (string memory) 
    {
        // Find first string element in array
        uint256 startIndex = 1; // Skip opening bracket
        while (startIndex < arrayJson.length && arrayJson[startIndex] != '"') {
            startIndex++;
        }
        
        if (startIndex >= arrayJson.length) return "";
        
        startIndex++; // Skip quote
        uint256 endIndex = startIndex;
        
        while (endIndex < arrayJson.length && arrayJson[endIndex] != '"') {
            endIndex++;
        }
        
        bytes memory result = new bytes(endIndex - startIndex);
        for (uint256 i = 0; i < endIndex - startIndex; i++) {
            result[i] = arrayJson[startIndex + i];
        }
        return string(result);
    }
    
    /**
     * @dev Parse capabilities from JSON array
     */
    function _parseCapabilities(string memory capabilitiesJson) 
        internal 
        pure 
        returns (bytes32[] memory) 
    {
        // Simplified parsing - count comma-separated items
        bytes memory jsonBytes = bytes(capabilitiesJson);
        uint256 count = 1;
        
        for (uint256 i = 0; i < jsonBytes.length; i++) {
            if (jsonBytes[i] == ',') count++;
        }
        
        bytes32[] memory capabilities = new bytes32[](count);
        
        // Extract each capability name (simplified)
        uint256 currentIndex = 0;
        // string memory currentCap = ""; // Removed unused variable
        
        for (uint256 i = 0; i < count && currentIndex < jsonBytes.length; i++) {
            // Find next string value
            while (currentIndex < jsonBytes.length && jsonBytes[currentIndex] != '"') {
                currentIndex++;
            }
            if (currentIndex < jsonBytes.length) {
                currentIndex++; // Skip quote
                uint256 endIndex = currentIndex;
                while (endIndex < jsonBytes.length && jsonBytes[endIndex] != '"') {
                    endIndex++;
                }
                
                bytes memory capBytes = new bytes(endIndex - currentIndex);
                for (uint256 j = 0; j < endIndex - currentIndex; j++) {
                    capBytes[j] = jsonBytes[currentIndex + j];
                }
                capabilities[i] = _stringToBytes32(string(capBytes));
                currentIndex = endIndex + 1;
            }
        }
        
        return capabilities;
    }
    
    /**
     * @dev Parse operations as capabilities
     */
    function _parseOperationsAsCapabilities(string memory operationsJson) 
        internal 
        pure 
        returns (bytes32[] memory) 
    {
        // Similar to _parseCapabilities but extract operation names
        return _parseCapabilities(operationsJson);
    }
    
    /**
     * @dev Escape JSON string
     */
    function _escapeJson(string memory input) internal pure returns (string memory) {
        bytes memory inputBytes = bytes(input);
        bytes memory escaped = new bytes(inputBytes.length * 2); // Max possible size
        uint256 escapedLength = 0;
        
        for (uint256 i = 0; i < inputBytes.length; i++) {
            bytes1 char = inputBytes[i];
            if (char == '"') {
                escaped[escapedLength++] = '\\';
                escaped[escapedLength++] = '"';
            } else if (char == '\\') {
                escaped[escapedLength++] = '\\';
                escaped[escapedLength++] = '\\';
            } else if (char == '\n') {
                escaped[escapedLength++] = '\\';
                escaped[escapedLength++] = 'n';
            } else if (char == '\r') {
                escaped[escapedLength++] = '\\';
                escaped[escapedLength++] = 'r';
            } else if (char == '\t') {
                escaped[escapedLength++] = '\\';
                escaped[escapedLength++] = 't';
            } else {
                escaped[escapedLength++] = char;
            }
        }
        
        // Create result with exact length
        bytes memory result = new bytes(escapedLength);
        for (uint256 i = 0; i < escapedLength; i++) {
            result[i] = escaped[i];
        }
        
        return string(result);
    }
    
    /**
     * @dev Generate message ID from content and type
     */
    function _generateMessageId(string memory content, bytes32 messageType) 
        internal 
        view 
        returns (string memory) 
    {
        bytes32 hash = keccak256(abi.encodePacked(content, messageType, block.timestamp));
        return _bytes32ToHex(hash);
    }
    
    /**
     * @dev Generate task ID from content and type
     */
    function _generateTaskId(string memory content, bytes32 messageType) 
        internal 
        view 
        returns (string memory) 
    {
        bytes32 hash = keccak256(abi.encodePacked("task", content, messageType, block.timestamp));
        return _bytes32ToHex(hash);
    }
    
    /**
     * @dev Get current timestamp as string
     */
    function _getCurrentTimestamp() internal view returns (string memory) {
        return _uint256ToString(block.timestamp);
    }
    
    /**
     * @dev Convert bytes32 to string
     */
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
    
    /**
     * @dev Convert string to bytes32
     */
    function _stringToBytes32(string memory source) internal pure returns (bytes32 result) {
        bytes memory tempEmptyStringTest = bytes(source);
        if (tempEmptyStringTest.length == 0) {
            return 0x0;
        }
        
        assembly {
            result := mload(add(source, 32))
        }
    }
    
    /**
     * @dev Convert uint256 to string
     */
    function _uint256ToString(uint256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        return string(buffer);
    }
    
    /**
     * @dev Convert bytes32 to hex string
     */
    function _bytes32ToHex(bytes32 _bytes32) internal pure returns (string memory) {
        bytes memory hexBytes = "0123456789abcdef";
        bytes memory result = new bytes(64);
        
        for (uint256 i = 0; i < 32; i++) {
            result[i * 2] = hexBytes[uint8(_bytes32[i] >> 4)];
            result[i * 2 + 1] = hexBytes[uint8(_bytes32[i] & 0x0f)];
        }
        
        return string(result);
    }
}