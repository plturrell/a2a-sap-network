// Location Data Standardization Framework
// Phase 1: Data Standardization Implementation

class LocationStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Load configuration with defaults
    this.config = this.loadConfiguration(config);
    
    // ISO 3166-1 Country Codes and Geographic Data
    this.countryMappings = this.config.countryMappings || {
      // Americas
      'United States': { iso2: 'US', iso3: 'USA', numeric: '840', lat: 39.8283, lng: -98.5795 },
      'Brazil': { iso2: 'BR', iso3: 'BRA', numeric: '076', lat: -14.2350, lng: -51.9253 },
      
      // Africa
      'Kenya': { iso2: 'KE', iso3: 'KEN', numeric: '404', lat: -0.0236, lng: 37.9062 },
      'Tanzania': { iso2: 'TZ', iso3: 'TZA', numeric: '834', lat: -6.3690, lng: 34.8888 },
      'Uganda': { iso2: 'UG', iso3: 'UGA', numeric: '800', lat: 1.3733, lng: 32.2903 },
      'Botswana': { iso2: 'BW', iso3: 'BWA', numeric: '072', lat: -22.3285, lng: 24.6849 },
      'Mauritius': { iso2: 'MU', iso3: 'MUS', numeric: '480', lat: -20.3484, lng: 57.5522 },
      'South Africa': { iso2: 'ZA', iso3: 'ZAF', numeric: '710', lat: -30.5595, lng: 22.9375 },
      'Zambia': { iso2: 'ZM', iso3: 'ZMB', numeric: '894', lat: -13.1339, lng: 27.8493 },
      'Nigeria': { iso2: 'NG', iso3: 'NGA', numeric: '566', lat: 9.0820, lng: 8.6753 },
      'Ghana': { iso2: 'GH', iso3: 'GHA', numeric: '288', lat: 7.9465, lng: -1.0232 },
      'Angola': { iso2: 'AO', iso3: 'AGO', numeric: '024', lat: -11.2027, lng: 17.8739 },
      'Cameroon': { iso2: 'CM', iso3: 'CMR', numeric: '120', lat: 7.3697, lng: 12.3547 },
      'Zimbabwe': { iso2: 'ZW', iso3: 'ZWE', numeric: '716', lat: -19.0154, lng: 29.1549 },
      'Sierra Leone': { iso2: 'SL', iso3: 'SLE', numeric: '694', lat: 8.4606, lng: -11.7799 },
      'Gambia': { iso2: 'GM', iso3: 'GMB', numeric: '270', lat: 13.4432, lng: -15.3101 },
      'Cote DIvoire': { iso2: 'CI', iso3: 'CIV', numeric: '384', lat: 7.5400, lng: -5.5471 },
      
      // Asia Pacific
      'China': { iso2: 'CN', iso3: 'CHN', numeric: '156', lat: 35.8617, lng: 104.1954 },
      'Japan': { iso2: 'JP', iso3: 'JPN', numeric: '392', lat: 36.2048, lng: 138.2529 },
      'Korea': { iso2: 'KR', iso3: 'KOR', numeric: '410', lat: 35.9078, lng: 127.7669 },
      'India': { iso2: 'IN', iso3: 'IND', numeric: '356', lat: 20.5937, lng: 78.9629 },
      'Singapore': { iso2: 'SG', iso3: 'SGP', numeric: '702', lat: 1.3521, lng: 103.8198 },
      'Malaysia': { iso2: 'MY', iso3: 'MYS', numeric: '458', lat: 4.2105, lng: 101.9758 },
      'Thailand': { iso2: 'TH', iso3: 'THA', numeric: '764', lat: 15.8700, lng: 100.9925 },
      'Indonesia': { iso2: 'ID', iso3: 'IDN', numeric: '360', lat: -0.7893, lng: 113.9213 },
      'Philippines': { iso2: 'PH', iso3: 'PHL', numeric: '608', lat: 12.8797, lng: 121.7740 },
      'Vietnam': { iso2: 'VN', iso3: 'VNM', numeric: '704', lat: 14.0583, lng: 108.2772 },
      'Taiwan': { iso2: 'TW', iso3: 'TWN', numeric: '158', lat: 23.6978, lng: 120.9605 },
      'Hong Kong': { iso2: 'HK', iso3: 'HKG', numeric: '344', lat: 22.3193, lng: 114.1694 },
      'Australia': { iso2: 'AU', iso3: 'AUS', numeric: '036', lat: -25.2744, lng: 133.7751 },
      'Bangladesh': { iso2: 'BD', iso3: 'BGD', numeric: '050', lat: 23.6850, lng: 90.3563 },
      'Pakistan': { iso2: 'PK', iso3: 'PAK', numeric: '586', lat: 30.3753, lng: 69.3451 },
      'Sri Lanka': { iso2: 'LK', iso3: 'LKA', numeric: '144', lat: 7.8731, lng: 80.7718 },
      'Nepal': { iso2: 'NP', iso3: 'NPL', numeric: '524', lat: 28.3949, lng: 84.1240 },
      'Cambodia': { iso2: 'KH', iso3: 'KHM', numeric: '116', lat: 12.5657, lng: 104.9910 },
      'Laos': { iso2: 'LA', iso3: 'LAO', numeric: '418', lat: 19.8563, lng: 102.4955 },
      'Myanmar': { iso2: 'MM', iso3: 'MMR', numeric: '104', lat: 21.9162, lng: 95.9560 },
      'Brunei': { iso2: 'BN', iso3: 'BRN', numeric: '096', lat: 4.5353, lng: 114.7277 },
      'Mongolia': { iso2: 'MN', iso3: 'MNG', numeric: '496', lat: 46.8625, lng: 103.8467 },
      
      // Europe
      'United Kingdom': { iso2: 'GB', iso3: 'GBR', numeric: '826', lat: 55.3781, lng: -3.4360 },
      'UK': { iso2: 'GB', iso3: 'GBR', numeric: '826', lat: 55.3781, lng: -3.4360 },
      'Germany': { iso2: 'DE', iso3: 'DEU', numeric: '276', lat: 51.1657, lng: 10.4515 },
      'France': { iso2: 'FR', iso3: 'FRA', numeric: '250', lat: 46.2276, lng: 2.2137 },
      'Ireland': { iso2: 'IE', iso3: 'IRL', numeric: '372', lat: 53.4129, lng: -8.2439 },
      'Luxembourg': { iso2: 'LU', iso3: 'LUX', numeric: '442', lat: 49.8153, lng: 6.1296 },
      'Switzerland': { iso2: 'CH', iso3: 'CHE', numeric: '756', lat: 46.8182, lng: 8.2275 },
      'Sweden': { iso2: 'SE', iso3: 'SWE', numeric: '752', lat: 60.1282, lng: 18.6435 },
      'Jersey': { iso2: 'JE', iso3: 'JEY', numeric: '832', lat: 49.2144, lng: -2.1312 },
      'Guernsey': { iso2: 'GG', iso3: 'GGY', numeric: '831', lat: 49.4658, lng: -2.5854 },
      'Falklands': { iso2: 'FK', iso3: 'FLK', numeric: '238', lat: -51.7963, lng: -59.5236 },
      
      // Middle East
      'UAE': { iso2: 'AE', iso3: 'ARE', numeric: '784', lat: 23.4241, lng: 53.8478 },
      'Qatar': { iso2: 'QA', iso3: 'QAT', numeric: '634', lat: 25.3548, lng: 51.1839 },
      'Bahrain': { iso2: 'BH', iso3: 'BHR', numeric: '048', lat: 26.0667, lng: 50.5577 },
      'Jordan': { iso2: 'JO', iso3: 'JOR', numeric: '400', lat: 30.5852, lng: 36.2384 },
      'Saudi Arabia': { iso2: 'SA', iso3: 'SAU', numeric: '682', lat: 23.8859, lng: 45.0792 },
      'Oman': { iso2: 'OM', iso3: 'OMN', numeric: '512', lat: 21.4735, lng: 55.9754 },
      'Lebanon': { iso2: 'LB', iso3: 'LBN', numeric: '422', lat: 33.8547, lng: 35.8623 },
      'Turkey': { iso2: 'TR', iso3: 'TUR', numeric: '792', lat: 38.9637, lng: 35.2433 },
      'Egypt': { iso2: 'EG', iso3: 'EGY', numeric: '818', lat: 26.8206, lng: 30.8025 },
      'Iraq': { iso2: 'IQ', iso3: 'IRQ', numeric: '368', lat: 33.2232, lng: 43.6793 }
    };

    // Regional center coordinates for aggregate regions
    this.regionalCenters = this.config.regionalCenters || {
      'Americas': { lat: 19.4326, lng: -99.1332 }, // Mexico City as Americas center
      'Africa': { lat: 0.0236, lng: 37.9062 }, // Kenya as Africa center
      'ASEAN': { lat: 1.3521, lng: 103.8198 }, // Singapore as ASEAN center
      'Europe': { lat: 50.1109, lng: 8.6821 }, // Frankfurt as Europe center
      'MENAP': { lat: 25.2048, lng: 55.2708 }, // Dubai as MENAP center
      'Greater China': { lat: 35.8617, lng: 104.1954 }, // China center
      'South Asia': { lat: 20.5937, lng: 78.9629 }, // India center
      'North East Asia': { lat: 36.2048, lng: 138.2529 } // Japan center
    };

    // UN M49 Regional codes
    this.unRegionalCodes = this.config.unRegionalCodes || {
      'Americas': '019',
      'Africa': '002', 
      'Asia': '142',
      'Europe': '150',
      'Oceania': '009'
    };
    
    // Processing statistics
    this.stats = {
      totalProcessed: 0,
      successfulStandardizations: 0,
      errors: 0,
      startTime: null,
      endTime: null
    };
  }

  // Create default logger
  createDefaultLogger() {
    return {
      info: (msg) => console.log(`[INFO] ${msg}`),
      warn: (msg) => console.warn(`[WARN] ${msg}`),
      error: (msg) => console.error(`[ERROR] ${msg}`),
      debug: (msg) => console.log(`[DEBUG] ${msg}`)
    };
  }

  // Load configuration with defaults
  loadConfiguration(config) {
    return {
      batchSize: 1000,
      enableValidation: true,
      logLevel: 'info',
      ...config
    };
  }

  // Clean internal codes and standardize names
  cleanLocationName(locationName) {
    try {
      if (!locationName) return '';
      
      return locationName
      .replace(/\s*-\s*Inp$/i, '')
      .replace(/\s*Other$/i, '')
      .replace(/^R\/o\s+/i, '')
      .replace(/\s*\(inc\s+[^)]+\)$/i, '')
      .replace(/_$/, '')
      .replace(/\s+/g, ' ')
      .trim();
    } catch (error) {
      this.logger.error(`Error cleaning location name: ${error.message}`);
      return locationName || '';
    }
  }

  // Get ISO codes and coordinates for a location
  getLocationStandardization(locationName) {
    try {
      const cleanName = this.cleanLocationName(locationName);
      
      // Try exact match first
      if (this.countryMappings[cleanName]) {
      return {
        cleanName,
        ...this.countryMappings[cleanName],
        locationType: 'country'
      };
    }

    // Try partial matches for common variations
    for (const [country, data] of Object.entries(this.countryMappings)) {
      if (cleanName.toLowerCase().includes(country.toLowerCase()) || 
          country.toLowerCase().includes(cleanName.toLowerCase())) {
        return {
          cleanName,
          ...data,
          locationType: 'country'
        };
      }
    }

    // Check if it's a regional grouping
    if (this.regionalCenters[cleanName]) {
      return {
        cleanName,
        iso2: null,
        iso3: null,
        numeric: null,
        ...this.regionalCenters[cleanName],
        locationType: 'region'
      };
    }

    // Special handling for financial centers and special zones
    const specialZones = {
      'Dubai Int Fin Centre': { lat: 25.2048, lng: 55.2708, parentCountry: 'UAE' },
      'ADGM': { lat: 24.4539, lng: 54.3773, parentCountry: 'UAE' },
      'DIFC': { lat: 25.2048, lng: 55.2708, parentCountry: 'UAE' },
      'Iran Kish': { lat: 26.5319, lng: 54.0167, parentCountry: 'IR' },
      'SC GBS Poland': { lat: 52.2297, lng: 21.0122, parentCountry: 'PL' }
    };

    if (specialZones[cleanName]) {
      return {
        cleanName,
        iso2: null,
        iso3: null,
        numeric: null,
        ...specialZones[cleanName],
        locationType: 'special_zone'
      };
    }

    // Return as-is with no standardization if no match found
    return {
      cleanName,
      iso2: null,
      iso3: null,
      numeric: null,
      lat: null,
      lng: null,
      locationType: 'unknown'
    };
    } catch (error) {
      this.logger.error(`Error getting location standardization: ${error.message}`);
      return {
        cleanName: locationName || '',
        iso2: null,
        iso3: null,
        numeric: null,
        lat: null,
        lng: null,
        locationType: 'error'
      };
    }
  }

  // Process a batch of records
  async processBatch(batch, startIndex) {
    const results = [];
    
    for (let i = 0; i < batch.length; i++) {
      const row = batch[i];
      
      try {
      const standardized = {};
      
      // Copy original data
      Object.keys(row).forEach(key => {
        standardized[`original_${key.replace(/[^a-zA-Z0-9]/g, '_')}`] = row[key];
      });

      // Add standardized location data for each level
      ['Location (L0)', 'Location (L1)', 'Location (L2)', 'Location (L3)', 'Location (L4)'].forEach((level, index) => {
        const levelNum = index;
        const locationData = this.getLocationStandardization(row[level]);
        
        standardized[`L${levelNum}_clean_name`] = locationData.cleanName;
        standardized[`L${levelNum}_iso2`] = locationData.iso2;
        standardized[`L${levelNum}_iso3`] = locationData.iso3;
        standardized[`L${levelNum}_numeric`] = locationData.numeric;
        standardized[`L${levelNum}_latitude`] = locationData.lat;
        standardized[`L${levelNum}_longitude`] = locationData.lng;
        standardized[`L${levelNum}_type`] = locationData.locationType;
      });

      // Add hierarchy path
      const cleanPath = [
        standardized.L0_clean_name,
        standardized.L1_clean_name,
        standardized.L2_clean_name,
        standardized.L3_clean_name,
        standardized.L4_clean_name
      ].filter(name => name && name.trim()).join(' → ');
      
      standardized.hierarchy_path = cleanPath;

      // Add primary coordinates (use most specific level available)
      for (let i = 4; i >= 0; i--) {
        if (standardized[`L${i}_latitude`] && standardized[`L${i}_longitude`]) {
          standardized.primary_latitude = standardized[`L${i}_latitude`];
          standardized.primary_longitude = standardized[`L${i}_longitude`];
          standardized.primary_location_level = i;
          break;
        }
      }

        results.push(standardized);
        this.stats.successfulStandardizations++;
      } catch (error) {
        this.logger.error(`Error processing row: ${error.message}`);
        this.stats.errors++;
        
        // Add error record
        results.push({
          original_row_number: row._row_number,
          error: error.message
        });
      }
    }
    
    return results;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting location standardization for ${data.length} records`);
    this.stats.startTime = Date.now();
    this.stats.totalProcessed = data.length;
    
    const results = [];
    const batchSize = this.config.batchSize;
    
    // Process in batches
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, Math.min(i + batchSize, data.length));
      const batchNumber = Math.floor(i / batchSize) + 1;
      const totalBatches = Math.ceil(data.length / batchSize);
      
      this.logger.info(`Processing batch ${batchNumber}/${totalBatches} (${batch.length} records)`);
      
      try {
        const batchResults = await this.processBatch(batch, i);
        results.push(...batchResults);
      } catch (error) {
        this.logger.error(`Error processing batch ${batchNumber}: ${error.message}`);
        this.stats.errors += batch.length;
      }
    }
    
    this.stats.endTime = Date.now();
    const duration = (this.stats.endTime - this.stats.startTime) / 1000;
    
    this.logger.info(`Location standardization completed in ${duration.toFixed(2)} seconds`);
    this.logger.info(`Successfully processed: ${this.stats.successfulStandardizations}/${this.stats.totalProcessed}`);
    if (this.stats.errors > 0) {
      this.logger.warn(`Errors encountered: ${this.stats.errors}`);
    }
    
    return results;
  }

  // Generate standardization report
  generateReport(originalData, standardizedData) {
    try {
      const report = {
      totalRecords: originalData.length,
      standardizationStats: {
        countriesIdentified: 0,
        regionsIdentified: 0,
        specialZonesIdentified: 0,
        unknownLocations: 0,
        coordinatesAdded: 0,
        isoCodesAdded: 0
      },
      qualityIssues: [],
      recommendations: []
    };

    standardizedData.forEach((row, index) => {
      // Count location types
      for (let i = 0; i <= 4; i++) {
        const locationType = row[`L${i}_type`];
        if (locationType === 'country') report.standardizationStats.countriesIdentified++;
        else if (locationType === 'region') report.standardizationStats.regionsIdentified++;
        else if (locationType === 'special_zone') report.standardizationStats.specialZonesIdentified++;
        else if (locationType === 'unknown') report.standardizationStats.unknownLocations++;
      }

      // Count coordinates and ISO codes added
      if (row.primary_latitude && row.primary_longitude) {
        report.standardizationStats.coordinatesAdded++;
      }
      
      if (row.L4_iso2 || row.L3_iso2) {
        report.standardizationStats.isoCodesAdded++;
      }

      // Identify quality issues
      if (!row.primary_latitude) {
        report.qualityIssues.push(`Row ${index + 1}: No coordinates available for "${row.hierarchy_path}"`);
      }
    });

    // Generate recommendations
    if (report.standardizationStats.unknownLocations > 0) {
      report.recommendations.push(`${report.standardizationStats.unknownLocations} locations need manual mapping to ISO standards`);
    }
    
    if (report.standardizationStats.coordinatesAdded < report.totalRecords) {
      report.recommendations.push(`${report.totalRecords - report.standardizationStats.coordinatesAdded} locations missing geographic coordinates`);
    }

    // Add processing statistics
    report.processingStats = {
      successfulStandardizations: this.stats.successfulStandardizations,
      errors: this.stats.errors,
      processingTime: this.stats.endTime ? ((this.stats.endTime - this.stats.startTime) / 1000).toFixed(2) + ' seconds' : 'N/A'
    };

    this.logger.info('Generated standardization report');
    return report;
    } catch (error) {
      this.logger.error(`Error generating report: ${error.message}`);
      return {
        error: error.message,
        totalRecords: originalData ? originalData.length : 0
      };
    }
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = LocationStandardizer;
} else {
  window.LocationStandardizer = LocationStandardizer;
}