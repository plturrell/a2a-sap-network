#!/usr/bin/env python3
"""
Test multi-pass Grok enrichment
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

async def test_multipass_enrichment():
    """Test multi-pass enrichment with difficult data"""
    
    print("Testing Multi-Pass Grok Enrichment")
    print("="*60)
    print(f"Grok API Available: {os.getenv('XAI_API_KEY') is not None}")
    print()
    
    # Test cases designed to need multiple passes
    test_cases = {
        "Location (Missing most data)": {
            "standardizer": LocationStandardizer(),
            "data": {"Location (L0)": "Europe", "Location (L1)": "Estonia", "_row_number": 1}
        },
        "Account (Vague description)": {
            "standardizer": AccountStandardizer(),
            "data": {"accountNumber": "9999", "accountDescription": "Miscellaneous", "_row_number": 1}
        },
        "Product (Incomplete hierarchy)": {
            "standardizer": ProductStandardizer(),
            "data": {"Product (L0)": "Structured Products", "_row_number": 1}
        }
    }
    
    for name, test_info in test_cases.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        try:
            standardizer = test_info["standardizer"]
            data = test_info["data"]
            
            # Enable debug logging to see enrichment passes
            import logging
            logging.getLogger('app.a2a.skills').setLevel(logging.INFO)
            
            # Run standardization
            result = await standardizer.standardize(data)
            
            if result.get("error"):
                print(f"✗ Error: {result['error']}")
            else:
                std = result.get("standardized", {})
                completeness = result.get("completeness", 0)
                metadata = result.get("metadata", {})
                enriched = metadata.get("enriched_with_ai", False)
                passes = metadata.get("enrichment_passes", 0)
                
                print(f"✓ Success")
                print(f"  Initial completeness: {completeness*100:.1f}%")
                print(f"  Enriched with AI: {enriched}")
                print(f"  Enrichment passes: {passes}")
                
                # Show what was enriched
                if name.startswith("Location"):
                    print(f"\n  Results after enrichment:")
                    print(f"    Name: {std.get('name')}")
                    print(f"    Country: {std.get('country')}")
                    print(f"    ISO2/ISO3: {std.get('iso2')}/{std.get('iso3')}")
                    print(f"    Region: {std.get('region')}")
                    print(f"    Subregion: {std.get('subregion')}")
                    coords = std.get('coordinates', {})
                    if coords:
                        print(f"    Coordinates: {coords.get('latitude')}, {coords.get('longitude')}")
                    
                elif name.startswith("Account"):
                    print(f"\n  Results after enrichment:")
                    print(f"    GL Code: {std.get('gl_account_code')}")
                    print(f"    Type: {std.get('account_type')}")
                    print(f"    Subtype: {std.get('account_subtype')}")
                    print(f"    Category: {std.get('account_category')}")
                    print(f"    IFRS9: {std.get('ifrs9_classification')}")
                    print(f"    Basel: {std.get('basel_classification')}")
                    
                elif name.startswith("Product"):
                    print(f"\n  Results after enrichment:")
                    print(f"    Code: {std.get('product_code')}")
                    print(f"    Category: {std.get('product_category')}")
                    print(f"    Family: {std.get('product_family')}")
                    print(f"    Basel: {std.get('basel_category')}")
                    print(f"    Regulatory: {std.get('regulatory_treatment')}")
                    print(f"    Risk: {std.get('risk_classification')}")
                
        except Exception as e:
            import traceback
            print(f"✗ Exception: {str(e)}")
            print(f"   {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(test_multipass_enrichment())