// Financial Services Product Standardization Framework
// Phase 1: Banking & Capital Markets Product Standardization

class ProductStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Load configuration with defaults
    this.config = this.loadConfiguration(config);
    
    // Financial services abbreviation mappings
    this.abbreviationMappings = this.config.abbreviationMappings || {
      // Capital Markets
      'CM': 'Capital Markets',
      'DCM': 'Debt Capital Markets',
      'CMS': 'Capital Markets Sales',
      'ECM': 'Equity Capital Markets',
      'M&A': 'Mergers and Acquisitions',
      'ESG': 'Environmental, Social, and Governance',
      
      // Trading & Markets
      'FX': 'Foreign Exchange',
      'FXO': 'Foreign Exchange Options',
      'FM': 'Financial Markets',
      'TM': 'Treasury Markets',
      'SPT': 'Structured Products Trading',
      'SCT': 'Structured Credit Trading',
      'XVA': 'X-Value Adjustments',
      'DVA': 'Debt Valuation Adjustment',
      'MTM': 'Mark to Market',
      'JV': 'Joint Venture',
      
      // Credit & Lending
      'CRE': 'Commercial Real Estate',
      'CCU': 'Central Credit Unit',
      'LAF': 'Leverage and Acquisition Financing',
      'IDFG': 'Infrastructure and Development Finance Group',
      'FITL': 'Finance and Investment Trading Limited',
      'ABL': 'Asset Based Lending',
      'SRL': 'Structured Rate Loan',
      'CLO': 'Collateralized Loan Obligation',
      'CDO': 'Collateralized Debt Obligation',
      
      // Transaction Banking
      'TS': 'Transaction Services',
      'TB': 'Transaction Banking',
      'TBFX': 'Transaction Banking Foreign Exchange',
      'CASA': 'Current Account Savings Account',
      'CFD': 'Central Funding Desk',
      'CMP': 'Cash Management Platform',
      
      // Securities Services
      'SS': 'Securities Services',
      'PS': 'Prime Services',
      'FIPB': 'Fixed Income Prime Brokerage',
      'NAV': 'Net Asset Value',
      
      // Geographic/Regional
      'AME': 'Americas',
      'EMEA': 'Europe, Middle East, and Africa',
      'GCNA': 'Greater China and North Asia',
      'EM': 'Emerging Markets',
      'DM': 'Developed Markets',
      'G10': 'Group of Ten',
      'LCY': 'Local Currency',
      
      // Operations & Support
      'FSS': 'Financial Services Support',
      'GCIG': 'Global Corporate and Investment Group',
      'PYOE': 'Prior Year Operational Events',
      'OR': 'Operational Risk',
      'AMS': 'Asset Management Services',
      'GBS': 'Global Business Services',
      'PMO': 'Portfolio Management Office',
      'PnL': 'Profit and Loss',
      'P&L': 'Profit and Loss',
      
      // Common terms
      'Inp': 'Input',
      'Inc': 'Including',
      'Exc': 'Excluding',
      'Mgmt': 'Management',
      'Mgt': 'Management',
      'Fin': 'Finance',
      'Ops': 'Operations',
      'Tech': 'Technology',
      'Corp': 'Corporate',
      'Govt': 'Government',
      'Dept': 'Department',
      'vs': 'versus',
      'w/': 'with',
      '&': 'and'
    };

    // Product classifications based on banking industry standards
    this.productClassifications = this.config.productClassifications || {
      // Investment Banking
      'Advisory': { 
        category: 'Investment Banking', 
        subcategory: 'Advisory Services', 
        revenue_type: 'Fee Income',
        business_line: 'Corporate and Investment Banking',
        regulatory_classification: 'Advisory'
      },
      'Capital Markets': { 
        category: 'Investment Banking', 
        subcategory: 'Capital Markets', 
        revenue_type: 'Fee Income',
        business_line: 'Corporate and Investment Banking',
        regulatory_classification: 'Underwriting'
      },
      'Mergers and Acquisitions': { 
        category: 'Investment Banking', 
        subcategory: 'M&A Advisory', 
        revenue_type: 'Fee Income',
        business_line: 'Corporate and Investment Banking',
        regulatory_classification: 'Advisory'
      },
      
      // Trading & Markets
      'Foreign Exchange': { 
        category: 'Markets', 
        subcategory: 'FX Trading', 
        revenue_type: 'Trading Income',
        business_line: 'Global Markets',
        regulatory_classification: 'Trading'
      },
      'Credit Trading': { 
        category: 'Markets', 
        subcategory: 'Credit Markets', 
        revenue_type: 'Trading Income',
        business_line: 'Global Markets',
        regulatory_classification: 'Trading'
      },
      'Rates': { 
        category: 'Markets', 
        subcategory: 'Rates Trading', 
        revenue_type: 'Trading Income',
        business_line: 'Global Markets',
        regulatory_classification: 'Trading'
      },
      'Commodities': { 
        category: 'Markets', 
        subcategory: 'Commodity Trading', 
        revenue_type: 'Trading Income',
        business_line: 'Global Markets',
        regulatory_classification: 'Trading'
      },
      
      // Lending & Credit
      'Lending': { 
        category: 'Banking', 
        subcategory: 'Corporate Lending', 
        revenue_type: 'Net Interest Income',
        business_line: 'Corporate and Investment Banking',
        regulatory_classification: 'Lending'
      },
      'Commercial Real Estate': { 
        category: 'Banking', 
        subcategory: 'Real Estate Finance', 
        revenue_type: 'Net Interest Income',
        business_line: 'Corporate and Investment Banking',
        regulatory_classification: 'Lending'
      },
      'Trade Finance': { 
        category: 'Banking', 
        subcategory: 'Trade Finance', 
        revenue_type: 'Fee Income',
        business_line: 'Transaction Banking',
        regulatory_classification: 'Trade Finance'
      },
      
      // Transaction Banking
      'Cash Management': { 
        category: 'Transaction Banking', 
        subcategory: 'Cash Management', 
        revenue_type: 'Fee Income',
        business_line: 'Transaction Banking',
        regulatory_classification: 'Transaction Services'
      },
      'Securities Services': { 
        category: 'Securities Services', 
        subcategory: 'Custody and Clearing', 
        revenue_type: 'Fee Income',
        business_line: 'Securities Services',
        regulatory_classification: 'Securities Services'
      },
      'Prime Services': { 
        category: 'Securities Services', 
        subcategory: 'Prime Brokerage', 
        revenue_type: 'Fee Income',
        business_line: 'Securities Services',
        regulatory_classification: 'Prime Brokerage'
      },
      
      // Support Functions
      'Risk Management': { 
        category: 'Support', 
        subcategory: 'Risk Management', 
        revenue_type: 'Cost Center',
        business_line: 'Risk and Control',
        regulatory_classification: 'Risk Management'
      },
      'Operations': { 
        category: 'Support', 
        subcategory: 'Operations', 
        revenue_type: 'Cost Center',
        business_line: 'Operations and Technology',
        regulatory_classification: 'Operations'
      }
    };

    // Geographic region mappings
    this.regionMappings = this.config.regionMappings || {
      'Americas': { region_code: 'AME', primary_currency: 'USD', time_zone: 'EST/EDT' },
      'EMEA': { region_code: 'EMEA', primary_currency: 'EUR', time_zone: 'GMT/CET' },
      'Greater China': { region_code: 'GCNA', primary_currency: 'CNY', time_zone: 'CST' },
      'Asia Pacific': { region_code: 'APAC', primary_currency: 'USD', time_zone: 'JST/SGT' },
      'Local Markets': { region_code: 'LOCAL', primary_currency: 'LOCAL', time_zone: 'LOCAL' }
    };

    // Business line hierarchies for validation
    this.businessLineHierarchy = this.config.businessLineHierarchy || {
      'Banking': ['Capital Markets & Advisory', 'Lending & Financing Solutions', 'Banking Others'],
      'Markets': ['Credit Trading', 'Macro Trading', 'Money Market', 'Securities & Prime Services'],
      'Transaction services': ['Cash', 'Trade & Working Capital', 'TS Others'],
      'Others': ['Others', 'Valuation & Other Adj']
    };

    // Revenue recognition categories
    this.revenueCategories = this.config.revenueCategories || {
      'Fee Income': 'Non-Interest Income',
      'Trading Income': 'Non-Interest Income', 
      'Net Interest Income': 'Interest Income',
      'Cost Center': 'Operating Expenses'
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

  // Clean and standardize product names
  cleanProductName(productName) {
    try {
      if (!productName) return '';
      
      let cleaned = productName
        .replace(/\s+/g, ' ')  // Remove extra spaces
        .trim();

    // Expand abbreviations
    Object.entries(this.abbreviationMappings).forEach(([abbr, expansion]) => {
      // Handle various abbreviation formats
      const patterns = [
        new RegExp(`\\b${abbr}\\b`, 'gi'),           // Standalone abbreviation
        new RegExp(`^${abbr}\\s`, 'gi'),             // Start of string
        new RegExp(`\\s${abbr}\\s`, 'gi'),           // Middle of string
        new RegExp(`\\s${abbr}$`, 'gi'),             // End of string
        new RegExp(`${abbr}-`, 'gi'),                // Hyphenated
        new RegExp(`${abbr}_`, 'gi'),                // Underscored
      ];
      
      patterns.forEach(pattern => {
        cleaned = cleaned.replace(pattern, (match) => {
          const prefix = match.startsWith(' ') ? ' ' : '';
          const suffix = match.endsWith(' ') || match.endsWith('-') || match.endsWith('_') ? 
                        match.slice(-1) : '';
          return `${prefix}${expansion}${suffix}`;
        });
      });
    });

    // Standardize common patterns
    cleaned = cleaned
      .replace(/\s*&\s*/g, ' and ')           // & → and
      .replace(/\s*\/\s*/g, ' / ')            // Standardize slashes
      .replace(/\s*-\s*/g, ' - ')             // Standardize hyphens
      .replace(/\(\s*/g, '(')                 // Remove space after (
      .replace(/\s*\)/g, ')')                 // Remove space before )
      .replace(/\s*_\s*/g, ' ')               // Remove underscores
      .replace(/\binput\b/gi, 'Input')        // Standardize "input"
      .replace(/\bother\b/gi, 'Other')        // Standardize "other"
      .replace(/\bothers\b/gi, 'Others')      // Standardize "others"
      .replace(/\btrading\b/gi, 'Trading')    // Standardize "trading"
      .replace(/\bservices\b/gi, 'Services')  // Standardize "services"
      .replace(/\bmanagement\b/gi, 'Management'); // Standardize "management"

    // Title case with financial services exceptions
    cleaned = cleaned.replace(/\b\w+/g, (word) => {
      // Keep certain words lowercase
      const lowercaseWords = ['and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'the', 'a', 'an', 'vs', 'versus'];
      // Keep certain abbreviations uppercase
      const uppercaseWords = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'FX', 'IR', 'EQ', 'FI', 'CO', 'CR'];
      
      if (lowercaseWords.includes(word.toLowerCase())) {
        return word.toLowerCase();
      } else if (uppercaseWords.includes(word.toUpperCase())) {
        return word.toUpperCase();
      } else {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
    });

    // Ensure first word is always capitalized
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);

    return cleaned;
    } catch (error) {
      this.logger.error(`Error cleaning product name: ${error.message}`);
      return productName || '';
    }
  }

  // Extract geographic region information
  extractRegion(productName) {
    try {
      const regions = Object.keys(this.regionMappings);
    for (const region of regions) {
      if (productName.toLowerCase().includes(region.toLowerCase())) {
        return {
          region: region,
          ...this.regionMappings[region]
        };
      }
    }
    
    // Check for region abbreviations in the mappings
    for (const [abbr, expansion] of Object.entries(this.abbreviationMappings)) {
      if (regions.includes(expansion) && productName.includes(abbr)) {
        return {
          region: expansion,
          ...this.regionMappings[expansion]
        };
      }
    }
    
    return {
      region: null,
      region_code: null,
      primary_currency: null,
      time_zone: null
    };
    } catch (error) {
      this.logger.error(`Error extracting region: ${error.message}`);
      return {
        region: null,
        region_code: null,
        primary_currency: null,
        time_zone: null
      };
    }
  }

  // Classify product type and business characteristics
  classifyProduct(cleanProductName, hierarchyPath) {
    try {
      // Try to match against known classifications
      for (const [pattern, classification] of Object.entries(this.productClassifications)) {
      if (cleanProductName.toLowerCase().includes(pattern.toLowerCase()) ||
          hierarchyPath.toLowerCase().includes(pattern.toLowerCase())) {
        return classification;
      }
    }

    // Fallback classification based on L1 category
    const pathParts = hierarchyPath.split(' → ');
    const l0 = pathParts[0];
    const l1 = pathParts[1];

    // Business line specific classifications
    if (l0 === 'Banking') {
      if (l1?.includes('Capital Markets')) {
        return {
          category: 'Investment Banking',
          subcategory: 'Capital Markets',
          revenue_type: 'Fee Income',
          business_line: 'Corporate and Investment Banking',
          regulatory_classification: 'Investment Banking'
        };
      } else if (l1?.includes('Lending')) {
        return {
          category: 'Banking',
          subcategory: 'Corporate Lending',
          revenue_type: 'Net Interest Income',
          business_line: 'Corporate and Investment Banking',
          regulatory_classification: 'Lending'
        };
      }
    } else if (l0 === 'Markets') {
      return {
        category: 'Markets',
        subcategory: 'Trading',
        revenue_type: 'Trading Income',
        business_line: 'Global Markets',
        regulatory_classification: 'Trading'
      };
    } else if (l0 === 'Transaction services') {
      return {
        category: 'Transaction Banking',
        subcategory: 'Transaction Services',
        revenue_type: 'Fee Income',
        business_line: 'Transaction Banking',
        regulatory_classification: 'Transaction Services'
      };
    }

    // Default classification
    return {
      category: 'Unknown',
      subcategory: 'Unknown',
      revenue_type: 'Unknown',
      business_line: 'Unknown',
      regulatory_classification: 'Unknown'
    };
    } catch (error) {
      this.logger.error(`Error classifying product: ${error.message}`);
      return {
        category: 'Unknown',
        subcategory: 'Unknown',
        revenue_type: 'Unknown',
        business_line: 'Unknown',
        regulatory_classification: 'Unknown'
      };
    }
  }

  // Generate product codes (simplified approach)
  generateProductCode(hierarchyPath, index) {
    try {
      const levels = hierarchyPath.split(' → ');
      const l0Code = this.getL0Code(levels[0]);
      const l1Code = this.getL1Code(levels[1]);
      const sequenceNumber = String(index + 1).padStart(3, '0');
      
      return `${l0Code}${l1Code}${sequenceNumber}`;
    } catch (error) {
      this.logger.error(`Error generating product code: ${error.message}`);
      return `UN00${String(index + 1).padStart(3, '0')}`;
    }
  }

  getL0Code(l0Name) {
    const codes = {
      'Banking': 'BK',
      'Markets': 'MK',
      'Transaction services': 'TS',
      'Others': 'OT'
    };
    return codes[l0Name] || 'UN';
  }

  getL1Code(l1Name) {
    if (!l1Name) return '00';
    // Create a simple hash for L1 codes
    let hash = 0;
    for (let i = 0; i < l1Name.length; i++) {
      hash = ((hash << 5) - hash + l1Name.charCodeAt(i)) & 0xfffffff;
    }
    return String(Math.abs(hash) % 100).padStart(2, '0');
  }

  // Validate hierarchy consistency
  validateHierarchy(l0, l1, l2, l3) {
    try {
      const issues = [];
      
      if (!this.config.enableValidation) {
        return issues;
      }
    
    // Check business line hierarchy rules
    if (this.businessLineHierarchy[l0] && !this.businessLineHierarchy[l0].includes(l1)) {
      issues.push(`L1 "${l1}" may not belong under L0 "${l0}"`);
    }
    
    // Check for repeated names across levels
    if (l0 === l1 && l1 === l2 && l2 === l3) {
      issues.push('All levels identical - may indicate incomplete hierarchy');
    }
    
    // Check for "Others" overuse
    const othersCount = [l0, l1, l2, l3].filter(level => 
      level?.toLowerCase().includes('others') || level?.toLowerCase().includes('other')).length;
    if (othersCount > 1) {
      issues.push('Multiple "Others" categories may indicate incomplete product definition');
    }
    
    return issues;
    } catch (error) {
      this.logger.error(`Error validating hierarchy: ${error.message}`);
      return [];
    }
  }

  // Process a batch of records
  async processBatch(batch, startIndex) {
    const results = [];
    
    for (let i = 0; i < batch.length; i++) {
      const row = batch[i];
      const index = startIndex + i;
      
      try {
      const result = {
        // Original data preserved
        original_row_number: row._row_number,
        original_L0: row['Product (L0)'],
        original_L1: row['Product (L1)'],
        original_L2: row['Product (L2)'],
        original_L3: row['Product (L3)']
      };

      // Process each hierarchical level
      const levels = [
        row['Product (L0)'],
        row['Product (L1)'],
        row['Product (L2)'],
        row['Product (L3)']
      ];

      levels.forEach((product, levelIndex) => {
        const cleanName = this.cleanProductName(product);
        const regionInfo = this.extractRegion(product);
        
        result[`L${levelIndex}_clean_name`] = cleanName;
        result[`L${levelIndex}_region`] = regionInfo.region;
        result[`L${levelIndex}_region_code`] = regionInfo.region_code;
        result[`L${levelIndex}_primary_currency`] = regionInfo.primary_currency;
        result[`L${levelIndex}_time_zone`] = regionInfo.time_zone;
      });

      // Create standardized hierarchy path
      result.hierarchy_path = levels
        .map(prod => this.cleanProductName(prod))
        .filter(name => name && name.trim())
        .join(' → ');

      // Product classification
      const classification = this.classifyProduct(result.L3_clean_name, result.hierarchy_path);
      result.product_category = classification.category;
      result.product_subcategory = classification.subcategory;
      result.revenue_type = classification.revenue_type;
      result.business_line = classification.business_line;
      result.regulatory_classification = classification.regulatory_classification;

      // Generate product code
      result.generated_product_code = this.generateProductCode(result.hierarchy_path, index);

      // Hierarchy validation
      const validationIssues = this.validateHierarchy(
        result.original_L0, result.original_L1, result.original_L2, result.original_L3
      );
      result.hierarchy_validation_issues = validationIssues.join('; ') || null;

      // Standardization flags
      result.has_regional_identifier = !!(result.L3_region || result.L2_region || result.L1_region || result.L0_region);
      result.has_currency_context = !!(result.L3_primary_currency || result.L2_primary_currency || result.L1_primary_currency);
      result.standardization_quality = this.assessStandardizationQuality(result);

        results.push(result);
        this.stats.successfulStandardizations++;
      } catch (error) {
        this.logger.error(`Error processing row ${row._row_number}: ${error.message}`);
        this.stats.errors++;
        
        // Add error record
        results.push({
          original_row_number: row._row_number,
          original_L0: row['Product (L0)'],
          original_L1: row['Product (L1)'],
          original_L2: row['Product (L2)'],
          original_L3: row['Product (L3)'],
          error: error.message,
          standardization_quality: 'Error'
        });
      }
    }
    
    return results;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting product standardization for ${data.length} records`);
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
    
    this.logger.info(`Product standardization completed in ${duration.toFixed(2)} seconds`);
    this.logger.info(`Successfully processed: ${this.stats.successfulStandardizations}/${this.stats.totalProcessed}`);
    if (this.stats.errors > 0) {
      this.logger.warn(`Errors encountered: ${this.stats.errors}`);
    }
    
    return results;
  }

  // Assess standardization quality
  assessStandardizationQuality(record) {
    try {
      let score = 0;
      let maxScore = 5;

    // Clean naming improvement (1 point)
    if (record.L3_clean_name !== record.original_L3) score += 1;

    // Product classification (1 point)
    if (record.product_category !== 'Unknown') score += 1;

    // Business line mapping (1 point)
    if (record.business_line !== 'Unknown') score += 1;

    // Regional/currency identification (1 point)
    if (record.has_regional_identifier || record.has_currency_context) score += 1;

    // Hierarchy consistency (1 point)
    if (!record.hierarchy_validation_issues) score += 1;

    const percentage = (score / maxScore) * 100;
    
    if (percentage >= 80) return 'Excellent';
    if (percentage >= 60) return 'Good';
    if (percentage >= 40) return 'Fair';
    return 'Needs Review';
    } catch (error) {
      this.logger.error(`Error assessing quality: ${error.message}`);
      return 'Unknown';
    }
  }
  // Generate standardization report
  generateReport(originalData, standardizedData) {
    try {
      const report = {
        totalRecords: originalData.length,
        processingStats: {
          successfulStandardizations: this.stats.successfulStandardizations,
          errors: this.stats.errors,
          processingTime: this.stats.endTime ? ((this.stats.endTime - this.stats.startTime) / 1000).toFixed(2) + ' seconds' : 'N/A'
        },
        standardizationStats: {
          productsWithRegionalIdentifier: standardizedData.filter(r => r.has_regional_identifier).length,
          productsWithCurrencyContext: standardizedData.filter(r => r.has_currency_context).length,
          productsClassified: standardizedData.filter(r => r.product_category !== 'Unknown').length,
          businessLinesIdentified: standardizedData.filter(r => r.business_line !== 'Unknown').length
        },
        qualityBreakdown: {
          excellent: standardizedData.filter(r => r.standardization_quality === 'Excellent').length,
          good: standardizedData.filter(r => r.standardization_quality === 'Good').length,
          fair: standardizedData.filter(r => r.standardization_quality === 'Fair').length,
          needsReview: standardizedData.filter(r => r.standardization_quality === 'Needs Review').length,
          errors: standardizedData.filter(r => r.standardization_quality === 'Error').length
        },
        productCategoryBreakdown: {},
        businessLineBreakdown: {},
        recommendations: []
      };

      // Product category breakdown
      standardizedData.forEach(record => {
        const category = record.product_category || 'Unknown';
        report.productCategoryBreakdown[category] = (report.productCategoryBreakdown[category] || 0) + 1;
        
        const businessLine = record.business_line || 'Unknown';
        report.businessLineBreakdown[businessLine] = (report.businessLineBreakdown[businessLine] || 0) + 1;
      });

      // Generate recommendations
      if (report.qualityBreakdown.needsReview > 0) {
        report.recommendations.push(`${report.qualityBreakdown.needsReview} products need manual review for complete standardization`);
      }
      
      if (report.qualityBreakdown.errors > 0) {
        report.recommendations.push(`${report.qualityBreakdown.errors} products encountered errors during processing`);
      }
      
      const unknownProducts = report.productCategoryBreakdown['Unknown'] || 0;
      if (unknownProducts > 0) {
        report.recommendations.push(`${unknownProducts} products could not be classified - review hierarchy mapping`);
      }

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
  module.exports = ProductStandardizer;
} else {
  window.ProductStandardizer = ProductStandardizer;
}