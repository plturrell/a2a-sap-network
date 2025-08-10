// Account Data Standardization Framework
// Phase 1: Banking/Financial Account Standardization

class AccountStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Performance tracking
    this.batchSize = config.batchSize || 1000;
    this.processingStats = {
      totalProcessed: 0,
      errors: 0,
      startTime: null,
      endTime: null
    };
    
    // Load configuration
    this.loadConfiguration(config);
  }
  
  // Create default logger if none provided
  createDefaultLogger() {
    return {
      info: (msg) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`),
      warn: (msg) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`),
      error: (msg) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`),
      debug: (msg) => console.debug(`[DEBUG] ${new Date().toISOString()} - ${msg}`)
    };
  }
  
  // Load configuration from external source or use defaults
  loadConfiguration(config) {
    try {
      // Load from config object or use defaults
      // Banking/Financial abbreviation mappings
      this.abbreviationMappings = config.abbreviationMappings || {
      'P/L': 'Profit and Loss',
      'RWA': 'Risk Weighted Assets',
      'ECL': 'Expected Credit Loss',
      'DVA': 'Debt Valuation Adjustment',
      'FVTPL': 'Fair Value Through Profit or Loss',
      'GIE': 'Gross Interest Expense',
      'APM': 'Alternative Performance Measure',
      'CEO': 'Chief Executive Officer',
      'CFO': 'Chief Financial Officer',
      'GBS': 'Global Business Services',
      'CCC': 'Central Cost Center',
      'TP': 'Transfer Pricing',
      'T&I': 'Technology and Infrastructure',
      'BAU': 'Business As Usual',
      'CDO': 'Collateralized Debt Obligation',
      'CLO': 'Collateralized Loan Obligation',
      'WRB': 'Wholesale Risk Banking',
      'NGA': 'Non-Group Assets',
      'NGL': 'Non-Group Liabilities',
      'Spec': 'Specific',
      'exc': 'excluding',
      'inc': 'including',
      'Av': 'Average',
      'Oth': 'Other',
      'Rev': 'Reverse',
      'Repo': 'Repurchase',
      'Liabs': 'Liabilities',
      'Chrg': 'Charge',
      'Serv': 'Service'
      };

      // Financial account classifications based on banking standards
      this.accountClassifications = config.accountClassifications || {
      // Asset Categories
      'Loans': { type: 'Asset', subtype: 'Credit Assets', ifrs9: 'Financial Assets', basel: 'Credit Risk' },
      'Derivative Assets': { type: 'Asset', subtype: 'Trading Assets', ifrs9: 'Financial Assets', basel: 'Market Risk' },
      'Investments': { type: 'Asset', subtype: 'Investment Securities', ifrs9: 'Financial Assets', basel: 'Market Risk' },
      'Non-Group Assets': { type: 'Asset', subtype: 'Other Assets', ifrs9: 'Non-Financial Assets', basel: 'Operational Risk' },
      
      // Liability Categories  
      'Deposits': { type: 'Liability', subtype: 'Customer Deposits', ifrs9: 'Financial Liabilities', basel: 'Credit Risk' },
      'Derivative Liabilities': { type: 'Liability', subtype: 'Trading Liabilities', ifrs9: 'Financial Liabilities', basel: 'Market Risk' },
      'Non-Group Liabilities': { type: 'Liability', subtype: 'Other Liabilities', ifrs9: 'Non-Financial Liabilities', basel: 'Operational Risk' },
      
      // Income Categories
      'Net Interest Income': { type: 'Income', subtype: 'Interest Income', ifrs9: 'Interest Revenue', basel: 'Net Interest Margin' },
      'Fee Income': { type: 'Income', subtype: 'Non-Interest Income', ifrs9: 'Fee Revenue', basel: 'Fee Income' },
      'Investment Recoveries': { type: 'Income', subtype: 'Other Income', ifrs9: 'Other Revenue', basel: 'Other Income' },
      
      // Expense Categories
      'Staff Costs': { type: 'Expense', subtype: 'Personnel Expenses', ifrs9: 'Operating Expenses', basel: 'Operating Costs' },
      'Legal Fees': { type: 'Expense', subtype: 'Professional Services', ifrs9: 'Operating Expenses', basel: 'Operating Costs' },
      'Premises': { type: 'Expense', subtype: 'Occupancy Costs', ifrs9: 'Operating Expenses', basel: 'Operating Costs' },
      'Training': { type: 'Expense', subtype: 'Staff Development', ifrs9: 'Operating Expenses', basel: 'Operating Costs' },
      
      // Provisions & Impairments
      'Credit Impairments': { type: 'Provision', subtype: 'Credit Loss Provision', ifrs9: 'Expected Credit Loss', basel: 'Credit Risk' },
      'Other Impairments': { type: 'Provision', subtype: 'Other Provisions', ifrs9: 'Impairment Loss', basel: 'Operational Risk' },
      
      // Risk Weighted Assets
      'Credit RWA': { type: 'Regulatory', subtype: 'Credit Risk Capital', ifrs9: 'N/A', basel: 'Credit Risk RWA' },
      'Market RWA': { type: 'Regulatory', subtype: 'Market Risk Capital', ifrs9: 'N/A', basel: 'Market Risk RWA' },
      'Operational RWA': { type: 'Regulatory', subtype: 'Operational Risk Capital', ifrs9: 'N/A', basel: 'Operational Risk RWA' }
      };

      // Currency/Region code mappings
      this.currencyRegionCodes = config.currencyRegionCodes || {
      '(0L)': { currency: 'USD', region: 'US', description: 'US Dollar' },
      '(ZM)': { currency: 'ZMW', region: 'ZM', description: 'Zambian Kwacha' },
      // Add more as needed based on your specific codes
      };

      // Account hierarchy validation rules
      this.hierarchyRules = config.hierarchyRules || {
      'Income': ['DVA', 'Fee Income', 'Interest Income', 'Other Income'],
      'Impairments': ['Credit Impairments', 'Other Impairments'],
      'Risk Weighted Assets': ['Credit RWA', 'Market RWA', 'Operational RWA'],
        'Total Cost': ['Staff Costs', 'Operating Costs', 'Other Costs']
      };
      
      this.logger.info('AccountStandardizer initialized successfully');
    } catch (error) {
      this.logger.error(`Failed to load configuration: ${error.message}`);
      throw error;
    }
  }

  // Clean and standardize account names
  cleanAccountName(accountName) {
    try {
      if (!accountName) return '';
    
    let cleaned = accountName
      .replace(/\s+/g, ' ')  // Remove extra spaces
      .trim();

    // Expand abbreviations
    Object.entries(this.abbreviationMappings).forEach(([abbr, expansion]) => {
      const regex = new RegExp(`\\b${abbr}\\b`, 'gi');
      cleaned = cleaned.replace(regex, expansion);
    });

    // Standardize common patterns
    cleaned = cleaned
      .replace(/\s*&\s*/g, ' and ')  // & → and
      .replace(/\s*\/\s*/g, ' / ')   // Standardize slashes
      .replace(/\(\s*/g, '(')        // Remove space after (
      .replace(/\s*\)/g, ')')        // Remove space before )
      .replace(/exc\s+/gi, 'excluding ')
      .replace(/inc\s+/gi, 'including ')
      .replace(/\bnet\b/gi, 'Net')
      .replace(/\bother\b/gi, 'Other')
      .replace(/\bspec\b/gi, 'Specific');

    // Capitalize first letter of each word (Title Case)
    cleaned = cleaned.replace(/\b\w+/g, (word) => {
      // Keep certain words lowercase
      const lowercaseWords = ['and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'the', 'a', 'an'];
      return lowercaseWords.includes(word.toLowerCase()) ? 
             word.toLowerCase() : 
             word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    });

      // Ensure first word is always capitalized
      cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);

      return cleaned;
    } catch (error) {
      this.logger.error(`Error cleaning account name '${accountName}': ${error.message}`);
      return accountName; // Return original on error
    }
  }

  // Extract currency/region information
  extractCurrencyRegion(accountName) {
    try {
      const matches = accountName.match(/\(([^)]+)\)/g);
    if (matches) {
      const code = matches[matches.length - 1]; // Get last parenthetical
      if (this.currencyRegionCodes[code]) {
        return {
          ...this.currencyRegionCodes[code],
          originalCode: code,
          accountNameWithoutCode: accountName.replace(code, '').trim()
        };
      }
    }
    return {
      currency: null,
      region: null,
      description: null,
      originalCode: null,
        accountNameWithoutCode: accountName
      };
    } catch (error) {
      this.logger.error(`Error extracting currency/region from '${accountName}': ${error.message}`);
      return {
        currency: null,
        region: null,
        description: null,
        originalCode: null,
        accountNameWithoutCode: accountName
      };
    }
  }

  // Classify account type
  classifyAccount(cleanAccountName, hierarchyPath) {
    // Try to match against known classifications
    for (const [pattern, classification] of Object.entries(this.accountClassifications)) {
      if (cleanAccountName.toLowerCase().includes(pattern.toLowerCase()) ||
          hierarchyPath.toLowerCase().includes(pattern.toLowerCase())) {
        return classification;
      }
    }

    // Fallback classification based on L0 category
    const l0 = hierarchyPath.split(' → ')[0];
    switch (l0.toLowerCase()) {
      case 'income':
        return { type: 'Income', subtype: 'Other Income', ifrs9: 'Other Revenue', basel: 'Other Income' };
      case 'impairments':
        return { type: 'Provision', subtype: 'Provisions', ifrs9: 'Impairment Loss', basel: 'Credit Risk' };
      case 'total cost':
        return { type: 'Expense', subtype: 'Operating Expenses', ifrs9: 'Operating Expenses', basel: 'Operating Costs' };
      case 'risk weighted assets':
        return { type: 'Regulatory', subtype: 'Risk Capital', ifrs9: 'N/A', basel: 'Risk Weighted Assets' };
      default:
        return { type: 'Unknown', subtype: 'Unknown', ifrs9: 'Unknown', basel: 'Unknown' };
    }
  }

  // Generate account codes (simplified approach)
  generateAccountCode(hierarchyPath, index) {
    const levels = hierarchyPath.split(' → ');
    const l0Code = this.getL0Code(levels[0]);
    const l1Code = this.getL1Code(levels[1]);
    const sequenceNumber = String(index + 1).padStart(3, '0');
    
    return `${l0Code}${l1Code}${sequenceNumber}`;
  }

  getL0Code(l0Name) {
    const codes = {
      'Impairments': '1',
      'Income': '2',
      'Net Interest Income': '3',
      'Total Cost': '4',
      'Risk Weighted Assets': '5',
      'Non-Group Assets': '6',
      'Non-Group Liabilities': '7',
      'Non-Group Contingents': '8'
    };
    return codes[l0Name] || '9';
  }

  getL1Code(l1Name) {
    // Simple hash-based approach for L1 codes
    if (!l1Name) return '00';
    let hash = 0;
    for (let i = 0; i < l1Name.length; i++) {
      hash = ((hash << 5) - hash + l1Name.charCodeAt(i)) & 0xfffffff;
    }
    return String(Math.abs(hash) % 100).padStart(2, '0');
  }

  // Validate hierarchy consistency
  validateHierarchy(l0, l1, l2, l3) {
    const issues = [];
    
    // Check for consistent naming across levels
    if (l0 === l1 && l1 === l2 && l2 === l3) {
      issues.push('All levels identical - may indicate data quality issue');
    }
    
    // Check hierarchy rules
    if (this.hierarchyRules[l0] && !this.hierarchyRules[l0].some(rule => 
      l1.toLowerCase().includes(rule.toLowerCase()))) {
      issues.push(`L1 "${l1}" may not belong under L0 "${l0}"`);
    }
    
    return issues;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting standardization of ${data.length} records`);
    this.processingStats.startTime = Date.now();
    this.processingStats.totalProcessed = 0;
    this.processingStats.errors = 0;
    
    const results = [];
    
    try {
      // Process in batches for better performance
      for (let i = 0; i < data.length; i += this.batchSize) {
        const batch = data.slice(i, Math.min(i + this.batchSize, data.length));
        const batchResults = await this.processBatch(batch, i);
        results.push(...batchResults);
        
        // Log progress
        this.processingStats.totalProcessed = results.length;
        if (i % (this.batchSize * 10) === 0) {
          this.logger.info(`Processed ${results.length}/${data.length} records`);
        }
      }
      
      this.processingStats.endTime = Date.now();
      const duration = (this.processingStats.endTime - this.processingStats.startTime) / 1000;
      this.logger.info(`Standardization completed: ${results.length} records in ${duration}s`);
      
      return results;
    } catch (error) {
      this.logger.error(`Fatal error during dataset standardization: ${error.message}`);
      throw error;
    }
  }
  
  // Process a batch of records
  async processBatch(batch, startIndex) {
    return batch.map((row, index) => {
      try {
      const result = {
        // Original data preserved
        original_row_number: row._row_number,
        original_L0: row['Account (L0)'],
        original_L1: row['Account (L1)'],
        original_L2: row['Account (L2)'],
        original_L3: row['Account (L3)']
      };

      // Process each hierarchical level
      const levels = [
        row['Account (L0)'],
        row['Account (L1)'],
        row['Account (L2)'],
        row['Account (L3)']
      ];

      levels.forEach((account, levelIndex) => {
        const cleanName = this.cleanAccountName(account);
        const currencyInfo = this.extractCurrencyRegion(account);
        
        result[`L${levelIndex}_clean_name`] = cleanName;
        result[`L${levelIndex}_currency`] = currencyInfo.currency;
        result[`L${levelIndex}_region`] = currencyInfo.region;
        result[`L${levelIndex}_currency_description`] = currencyInfo.description;
        result[`L${levelIndex}_name_without_currency`] = this.cleanAccountName(currencyInfo.accountNameWithoutCode);
      });

      // Create standardized hierarchy path
      result.hierarchy_path = levels
        .map(acc => this.cleanAccountName(acc))
        .filter(name => name && name.trim())
        .join(' → ');

      // Account classification
      const classification = this.classifyAccount(result.L3_clean_name, result.hierarchy_path);
      result.account_type = classification.type;
      result.account_subtype = classification.subtype;
      result.ifrs9_classification = classification.ifrs9;
      result.basel_classification = classification.basel;

      // Generate account code
      result.generated_account_code = this.generateAccountCode(result.hierarchy_path, index);

      // Hierarchy validation
      const validationIssues = this.validateHierarchy(
        result.original_L0, result.original_L1, result.original_L2, result.original_L3
      );
      result.hierarchy_validation_issues = validationIssues.join('; ') || null;

      // Standardization flags
      result.has_currency_code = !!(result.L3_currency || result.L2_currency || result.L1_currency);
      result.has_regional_code = !!(result.L3_region || result.L2_region || result.L1_region);
      result.standardization_quality = this.assessStandardizationQuality(result);

        return result;
      } catch (error) {
        this.processingStats.errors++;
        this.logger.error(`Error processing row ${startIndex + index}: ${error.message}`);
        
        // Return partial result with error flag
        return {
          original_row_number: row._row_number || startIndex + index,
          error: error.message,
          standardization_quality: 'Error'
        };
      }
    });
  }

  // Assess standardization quality
  assessStandardizationQuality(record) {
    let score = 0;
    let maxScore = 5;

    // Clean naming (1 point)
    if (record.L3_clean_name !== record.original_L3) score += 1;

    // Account classification (1 point)
    if (record.account_type !== 'Unknown') score += 1;

    // Currency/region identification (1 point)
    if (record.has_currency_code || record.has_regional_code) score += 1;

    // Hierarchy consistency (1 point)
    if (!record.hierarchy_validation_issues) score += 1;

    // Complete standardization (1 point)
    if (record.ifrs9_classification !== 'Unknown' && record.basel_classification !== 'Unknown') score += 1;

    const percentage = (score / maxScore) * 100;
    
    if (percentage >= 80) return 'Excellent';
    if (percentage >= 60) return 'Good';
    if (percentage >= 40) return 'Fair';
    return 'Needs Review';
  }

  // Generate standardization report
  generateReport(originalData, standardizedData) {
    try {
      const report = {
      totalRecords: originalData.length,
      standardizationStats: {
        accountsWithCurrencyCodes: standardizedData.filter(r => r.has_currency_code).length,
        accountsWithRegionalCodes: standardizedData.filter(r => r.has_regional_code).length,
        accountsClassified: standardizedData.filter(r => r.account_type !== 'Unknown').length,
        hierarchyIssues: standardizedData.filter(r => r.hierarchy_validation_issues).length
      },
      qualityBreakdown: {
        excellent: standardizedData.filter(r => r.standardization_quality === 'Excellent').length,
        good: standardizedData.filter(r => r.standardization_quality === 'Good').length,
        fair: standardizedData.filter(r => r.standardization_quality === 'Fair').length,
        needsReview: standardizedData.filter(r => r.standardization_quality === 'Needs Review').length
      },
      accountTypeBreakdown: {},
      recommendations: []
    };

    // Account type breakdown
    standardizedData.forEach(record => {
      const type = record.account_type;
      report.accountTypeBreakdown[type] = (report.accountTypeBreakdown[type] || 0) + 1;
    });

    // Generate recommendations
    if (report.standardizationStats.hierarchyIssues > 0) {
      report.recommendations.push(`${report.standardizationStats.hierarchyIssues} accounts have hierarchy validation issues that need review`);
    }
    
    if (report.qualityBreakdown.needsReview > 0) {
      report.recommendations.push(`${report.qualityBreakdown.needsReview} accounts need manual review for complete standardization`);
    }

      // Add processing statistics
      report.processingStats = {
        ...this.processingStats,
        recordsPerSecond: this.processingStats.totalProcessed / 
          ((this.processingStats.endTime - this.processingStats.startTime) / 1000)
      };
      
      return report;
    } catch (error) {
      this.logger.error(`Error generating report: ${error.message}`);
      return { error: error.message };
    }
  }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AccountStandardizer;
} else {
  window.AccountStandardizer = AccountStandardizer;
}