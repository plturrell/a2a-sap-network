#!/usr/bin/env python3
"""
Test standardizers directly without agent
"""

import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Import standardizers
from app.a2a.skills.location_standardizer import LocationStandardizer
from app.a2a.skills.account_standardizer import AccountStandardizer
from app.a2a.skills.product_standardizer import ProductStandardizer
from app.a2a.skills.book_standardizer import BookStandardizer
from app.a2a.skills.measure_standardizer import MeasureStandardizer

async def test_standardizers():
    """Test each standardizer directly"""
    
    print("Testing Standardizers Directly")
    print("="*60)
    print(f"Grok API Available: {os.getenv('XAI_API_KEY') is not None}")
    print()
    
    # Test data
    test_cases = {
        "Location": {
            "standardizer": LocationStandardizer(),
            "data": {"Location (L0)": "Americas", "Location (L1)": "Mexico", "Location (L2)": "Mexico City", "_row_number": 1}
        },
        "Account": {
            "standardizer": AccountStandardizer(),
            "data": {"accountNumber": "4001", "accountDescription": "Revenue - Trading", "costCenter": "CC400", "_row_number": 1}
        },
        "Product": {
            "standardizer": ProductStandardizer(),
            "data": {"Product (L0)": "Commodities", "Product (L1)": "Energy", "Product (L2)": "Oil Futures", "_row_number": 1}
        },
        "Book": {
            "standardizer": BookStandardizer(),
            "data": {"Books": "Investment Bank - Manual Adjustment", "_row_number": 1}
        },
        "Measure": {
            "standardizer": MeasureStandardizer(),
            "data": {"measureType": "Forecast", "Version": "Q2", "Currency": "RFX", "_row_number": 1}
        }
    }
    
    for name, test_info in test_cases.items():
        print(f"\n{name} Standardization:")
        print("-" * 40)
        
        try:
            standardizer = test_info["standardizer"]
            data = test_info["data"]
            
            # Run standardization
            result = await standardizer.standardize(data)
            
            if result.get("error"):
                print(f"✗ Error: {result['error']}")
            else:
                std = result.get("standardized", {})
                completeness = result.get("completeness", 0)
                enriched = result.get("metadata", {}).get("enriched_with_ai", False)
                
                print(f"✓ Success")
                print(f"  Completeness: {completeness*100:.1f}%")
                print(f"  Enriched with AI: {enriched}")
                
                # Show key fields
                if name == "Location":
                    print(f"  Name: {std.get('name')}")
                    print(f"  ISO: {std.get('iso2')}/{std.get('iso3')}")
                elif name == "Account":
                    print(f"  GL Code: {std.get('gl_account_code')}")
                    print(f"  Category: {std.get('account_category')}")
                elif name == "Product":
                    print(f"  Category: {std.get('product_category')}")
                    print(f"  Basel: {std.get('basel_category')}")
                elif name == "Book":
                    print(f"  Entity Type: {std.get('entity_type')}")
                    print(f"  Book Type: {std.get('book_type')}")
                elif name == "Measure":
                    print(f"  Type: {std.get('measure_type')}")
                    print(f"  Category: {std.get('category')}")
                
        except Exception as e:
            import traceback
            print(f"✗ Exception: {str(e)}")
            print(f"   {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_standardizers())