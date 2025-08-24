# API Request Signing Documentation

## Overview

The A2A platform supports HMAC-based request signing for enhanced API security. Request signing provides:

- **Authentication**: Verify the identity of the API client
- **Integrity**: Ensure request data hasn't been tampered with
- **Non-repudiation**: Requests can be traced to specific API keys
- **Replay protection**: Prevent replay attacks using timestamps and nonces

## How It Works

### 1. Signature Generation

Requests are signed using HMAC-SHA256 over a canonical representation of:
- HTTP method
- Request path
- Timestamp
- Nonce (unique request identifier)
- Query parameters (sorted)
- Body hash (for requests with bodies)

### 2. Required Headers

Signed requests must include these headers:

```
X-API-Key-ID: your_api_key_id
X-Timestamp: 1703123456
X-Nonce: abc123def456
X-Signature: base64_encoded_signature
X-Body-Hash: base64_encoded_sha256_hash (if body present)
```

### 3. Timestamp Validation

Requests must be made within 5 minutes of the timestamp to be considered valid.

### 4. Nonce Tracking

Each nonce can only be used once to prevent replay attacks.

## API Key Management

### Creating an API Key

```bash
curl -X POST http://localhost:8000/api/v1/admin/api-keys \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "key_id": "my-service-key",
    "permissions": ["read", "write"],
    "description": "Key for my service integration"
  }'
```

Response:
```json
{
  "key_id": "my-service-key",
  "secret": "base64_encoded_secret_key",
  "active": true,
  "permissions": ["read", "write"],
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Important**: Save the secret immediately - it won't be shown again!

### Available Permissions

- `read`: Read access to resources
- `write`: Write/modify access to resources
- `admin`: Administrative operations
- `a2a`: Agent-to-agent communication

### Rotating an API Key

```bash
curl -X POST http://localhost:8000/api/v1/admin/api-keys/my-service-key/rotate \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Revoking an API Key

```bash
curl -X DELETE http://localhost:8000/api/v1/admin/api-keys/my-service-key \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

## Implementation Examples

### Python Client Example

```python
import hmac
import hashlib
import time
import base64
import json
import httpx

class SignedAPIClient:
    def __init__(self, base_url, api_key_id, api_secret):
        self.base_url = base_url
        self.api_key_id = api_key_id
        self.api_secret = api_secret
    
    def _generate_signature(self, method, path, timestamp, nonce, body_hash=None):
        # Build canonical string
        parts = [method.upper(), path, str(timestamp), nonce]
        if body_hash:
            parts.append(body_hash)
        
        canonical = "\n".join(parts)
        
        # Generate HMAC
        signature = hmac.new(
            self.api_secret.encode(),
            canonical.encode(),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode()
    
    async def request(self, method, path, json_data=None):
        # Generate signing components
        timestamp = int(time.time())
        nonce = base64.b64encode(os.urandom(16)).decode()[:16]
        
        # Calculate body hash if needed
        body_hash = None
        body = None
        if json_data:
            body = json.dumps(json_data).encode()
            body_hash = base64.b64encode(
                hashlib.sha256(body).digest()
            ).decode()
        
        # Generate signature
        signature = self._generate_signature(
            method, path, timestamp, nonce, body_hash
        )
        
        # Make request
        headers = {
            "X-API-Key-ID": self.api_key_id,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "X-Signature": signature
        }
        
        if body_hash:
            headers["X-Body-Hash"] = body_hash
            headers["Content-Type"] = "application/json"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                content=body
            )
            return response
```

### JavaScript/Node.js Example

```javascript
const crypto = require('crypto');
const axios = require('axios');

class SignedAPIClient {
  constructor(baseURL, apiKeyId, apiSecret) {
    this.baseURL = baseURL;
    this.apiKeyId = apiKeyId;
    this.apiSecret = apiSecret;
  }
  
  generateSignature(method, path, timestamp, nonce, bodyHash) {
    // Build canonical string
    const parts = [method.toUpperCase(), path, timestamp, nonce];
    if (bodyHash) {
      parts.push(bodyHash);
    }
    
    const canonical = parts.join('\n');
    
    // Generate HMAC
    const hmac = crypto.createHmac('sha256', this.apiSecret);
    hmac.update(canonical);
    
    return hmac.digest('base64');
  }
  
  async request(method, path, data = null) {
    // Generate signing components
    const timestamp = Math.floor(Date.now() / 1000);
    const nonce = crypto.randomBytes(8).toString('base64').substring(0, 16);
    
    // Calculate body hash if needed
    let bodyHash = null;
    let body = null;
    
    if (data) {
      body = JSON.stringify(data);
      const hash = crypto.createHash('sha256');
      hash.update(body);
      bodyHash = hash.digest('base64');
    }
    
    // Generate signature
    const signature = this.generateSignature(
      method, path, timestamp, nonce, bodyHash
    );
    
    // Prepare headers
    const headers = {
      'X-API-Key-ID': this.apiKeyId,
      'X-Timestamp': timestamp.toString(),
      'X-Nonce': nonce,
      'X-Signature': signature
    };
    
    if (bodyHash) {
      headers['X-Body-Hash'] = bodyHash;
      headers['Content-Type'] = 'application/json';
    }
    
    // Make request
    const response = await axios({
      method,
      url: `${this.baseURL}${path}`,
      headers,
      data: body
    });
    
    return response;
  }
}
```

## Security Considerations

1. **Secret Storage**: Never store API secrets in code or version control
2. **HTTPS Only**: Always use HTTPS in production to prevent man-in-the-middle attacks
3. **Key Rotation**: Regularly rotate API keys, especially if compromise is suspected
4. **Least Privilege**: Grant only the minimum permissions needed
5. **Monitoring**: Monitor API key usage for suspicious patterns

## Troubleshooting

### Common Errors

1. **"Missing required signing headers"**
   - Ensure all required headers are present
   - Check header names are correct (case-insensitive)

2. **"Request timestamp is too old"**
   - Ensure client clock is synchronized
   - Request must be made within 5 minutes of timestamp

3. **"Nonce has already been used"**
   - Generate a new unique nonce for each request
   - Don't reuse nonces from previous requests

4. **"Invalid signature"**
   - Verify canonical string construction matches server
   - Check API secret is correct
   - Ensure body hash calculation is correct

5. **"Body integrity check failed"**
   - Body hash must be calculated over exact bytes sent
   - Don't modify body after hash calculation

## API Endpoints

### API Key Management

- `POST /api/v1/admin/api-keys` - Create new API key
- `GET /api/v1/admin/api-keys` - List all API keys
- `GET /api/v1/admin/api-keys/{key_id}` - Get specific API key details
- `POST /api/v1/admin/api-keys/{key_id}/rotate` - Rotate API key secret
- `DELETE /api/v1/admin/api-keys/{key_id}` - Revoke API key

## Best Practices

1. **Use request signing for**:
   - Service-to-service communication
   - Automated scripts and bots
   - High-security operations
   - Audit trail requirements

2. **Combine with user authentication**:
   - Request signing verifies the client application
   - Bearer tokens authenticate the end user
   - Use both for maximum security

3. **Monitor and audit**:
   - Track API key usage patterns
   - Set up alerts for suspicious activity
   - Regular review of active keys
   - Remove unused keys promptly