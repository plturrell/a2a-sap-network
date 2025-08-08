#!/usr/bin/env python3
"""
Simple HANA connection test to verify credentials
"""

import os
from dotenv import load_dotenv

# Load environment variables from project root
load_dotenv("/Users/apple/projects/finsight_cib/.env")

def test_hana_connection():
    """Test basic HANA connection"""
    try:
        from hdbcli import dbapi
        
        print("Testing SAP HANA Cloud connection...")
        print(f"Host: {os.getenv('HANA_HOSTNAME')}")
        print(f"Port: {os.getenv('HANA_PORT')}")
        print(f"User: {os.getenv('HANA_USERNAME')}")
        print("Password: [REDACTED]")
        
        # Connect to HANA
        connection = dbapi.connect(
            address=os.getenv('HANA_HOSTNAME'),
            port=int(os.getenv('HANA_PORT')),
            user=os.getenv('HANA_USERNAME'),
            password=os.getenv('HANA_PASSWORD'),
            encrypt=True,
            sslValidateCertificate=False
        )
        
        print("✓ Successfully connected to SAP HANA Cloud!")
        
        # Test basic query
        cursor = connection.cursor()
        cursor.execute("SELECT 'Hello HANA!' FROM DUMMY")
        result = cursor.fetchone()
        print(f"✓ Test query result: {result[0]}")
        
        # Check Vector Engine capability
        cursor.execute("SELECT DATABASE_NAME, SQL_PORT FROM M_DATABASES")
        results = cursor.fetchall()
        print(f"✓ Connected to database: {results}")
        
        cursor.close()
        connection.close()
        print("✓ Connection test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_hana_connection()