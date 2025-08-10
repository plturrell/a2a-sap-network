// Test JavaScript standardizer directly
const LocationStandardizer = require('../scripts/build/location_standardization.js');

async function test() {
    console.log('Testing Location Standardizer...');
    
    const standardizer = new LocationStandardizer();
    
    const testData = [
        {
            "Location (L0)": "Americas",
            "Location (L1)": "United States", 
            "Location (L2)": "New York",
            "Location (L3)": "New York City",
            "Location (L4)": "Manhattan",
            "_row_number": 1
        }
    ];
    
    console.log('Input data:', JSON.stringify(testData, null, 2));
    
    try {
        // standardizeDataset is async
        const result = await standardizer.standardizeDataset(testData);
        console.log('Result type:', typeof result);
        console.log('Result length:', result ? result.length : 'null');
        console.log('Result:', JSON.stringify(result, null, 2));
        
        if (result && result.length > 0) {
            console.log('First item:', result[0]);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

test();