// Financial Measurement Configuration Standardization Framework
// Phase 1: Reporting Dimension & Currency Treatment Standardization

class MeasureStandardizer {
  constructor(config = {}) {
    // Initialize logger
    this.logger = config.logger || this.createDefaultLogger();
    
    // Load configuration with defaults
    this.config = this.loadConfiguration(config);
    
    // Financial measurement abbreviation mappings
    this.abbreviationMappings = this.config.abbreviationMappings || {
      // Version/Period abbreviations
      'MTD': 'Month-to-Date',
      'YTD': 'Year-to-Date',
      'QTD': 'Quarter-to-Date',
      'WTD': 'Week-to-Date',
      'ITD': 'Inception-to-Date',
      'PTD': 'Period-to-Date',
      
      // Measure type abbreviations
      'Act': 'Actual',
      'Bud': 'Budget',
      'Fcst': 'Forecast',
      'Est': 'Estimate',
      'Plan': 'Plan',
      'Proj': 'Projection',
      'Tgt': 'Target',
      'Bench': 'Benchmark',
      'Prior': 'Prior Period',
      'Var': 'Variance',
      
      // Currency type abbreviations
      'CFX': 'Constant Foreign Exchange',
      'RFX': 'Reporting Foreign Exchange',
      'FWD': 'Forward Foreign Exchange',
      'SPT': 'Spot Foreign Exchange',
      'AVG': 'Average Foreign Exchange',
      'EOD': 'End of Day Foreign Exchange',
      'EOM': 'End of Month Foreign Exchange',
      'EOY': 'End of Year Foreign Exchange',
      'HED': 'Hedged Foreign Exchange',
      'UHD': 'Unhedged Foreign Exchange',
      'LOC': 'Local Currency',
      'REP': 'Reporting Currency',
      'FUN': 'Functional Currency',
      'TRN': 'Transaction Currency'
    };

    // Measurement type classifications
    this.measurementClassifications = this.config.measurementClassifications || {
      // Actual measurements
      'Actual': {
        category: 'Historical',
        subcategory: 'Actual Results',
        data_source: 'General Ledger',
        timing: 'Post-Event',
        accuracy: 'High',
        regulatory_usage: 'Primary',
        audit_requirement: 'Full Audit'
      },
      
      // Planning measurements
      'Budget': {
        category: 'Planning',
        subcategory: 'Annual Budget',
        data_source: 'Planning System',
        timing: 'Pre-Event',
        accuracy: 'Medium',
        regulatory_usage: 'Supporting',
        audit_requirement: 'Review'
      },
      'Forecast': {
        category: 'Planning',
        subcategory: 'Dynamic Forecast',
        data_source: 'Planning System',
        timing: 'Pre-Event',
        accuracy: 'Medium',
        regulatory_usage: 'Supporting',
        audit_requirement: 'Review'
      },
      'Plan': {
        category: 'Planning',
        subcategory: 'Strategic Plan',
        data_source: 'Planning System',
        timing: 'Pre-Event',
        accuracy: 'Low',
        regulatory_usage: 'Internal',
        audit_requirement: 'None'
      },
      
      // Analytical measurements
      'Variance': {
        category: 'Analysis',
        subcategory: 'Variance Analysis',
        data_source: 'Calculated',
        timing: 'Post-Event',
        accuracy: 'High',
        regulatory_usage: 'Supporting',
        audit_requirement: 'Review'
      },
      'Prior Period': {
        category: 'Comparative',
        subcategory: 'Historical Comparison',
        data_source: 'General Ledger',
        timing: 'Post-Event',
        accuracy: 'High',
        regulatory_usage: 'Supporting',
        audit_requirement: 'Full Audit'
      }
    };

    // Currency treatment classifications
    this.currencyClassifications = this.config.currencyClassifications || {
      // FX Rate Types
      'Constant Foreign Exchange': {
        fx_treatment: 'Fixed Rate',
        purpose: 'Performance Analysis',
        volatility_impact: 'Eliminated',
        usage: 'Management Reporting',
        calculation_method: 'Fixed Historical Rate',
        regulatory_compliance: 'Internal Use'
      },
      'Reporting Foreign Exchange': {
        fx_treatment: 'Current Rate',
        purpose: 'Financial Reporting',
        volatility_impact: 'Included',
        usage: 'External Reporting',
        calculation_method: 'Current Market Rate',
        regulatory_compliance: 'IFRS/GAAP Required'
      },
      'Forward Foreign Exchange': {
        fx_treatment: 'Forward Rate',
        purpose: 'Risk Management',
        volatility_impact: 'Hedged',
        usage: 'Treasury Reporting',
        calculation_method: 'Forward Contract Rate',
        regulatory_compliance: 'Hedge Accounting'
      },
      'Spot Foreign Exchange': {
        fx_treatment: 'Spot Rate',
        purpose: 'Current Valuation',
        volatility_impact: 'Current Market',
        usage: 'Trading Reporting',
        calculation_method: 'Current Spot Rate',
        regulatory_compliance: 'Mark-to-Market'
      },
      'Average Foreign Exchange': {
        fx_treatment: 'Average Rate',
        purpose: 'Smoothed Reporting',
        volatility_impact: 'Smoothed',
        usage: 'Trend Analysis',
        calculation_method: 'Period Average Rate',
        regulatory_compliance: 'Alternative Measure'
      }
    };

    // Version/Period classifications
    this.versionClassifications = this.config.versionClassifications || {
      'Month-to-Date': {
        period_type: 'Cumulative',
        granularity: 'Monthly',
        reset_frequency: 'Monthly',
        typical_usage: 'Operational Reporting',
        comparison_basis: 'Prior Month MTD'
      },
      'Year-to-Date': {
        period_type: 'Cumulative',
        granularity: 'Annual',
        reset_frequency: 'Annually',
        typical_usage: 'Financial Reporting',
        comparison_basis: 'Prior Year YTD'
      },
      'Quarter-to-Date': {
        period_type: 'Cumulative',
        granularity: 'Quarterly',
        reset_frequency: 'Quarterly',
        typical_usage: 'Earnings Reporting',
        comparison_basis: 'Prior Quarter QTD'
      }
    };

    // Reporting frameworks
    this.reportingFrameworks = this.config.reportingFrameworks || {
      'IFRS': {
        framework: 'International Financial Reporting Standards',
        measurement_focus: 'Fair Value',
        currency_requirements: 'Functional Currency Primary',
        period_requirements: 'Comparative Periods'
      },
      'GAAP': {
        framework: 'Generally Accepted Accounting Principles',
        measurement_focus: 'Historical Cost',
        currency_requirements: 'Reporting Currency',
        period_requirements: 'Consistent Periods'
      },
      'Basel': {
        framework: 'Basel Regulatory Framework',
        measurement_focus: 'Risk Measurement',
        currency_requirements: 'Home Currency',
        period_requirements: 'Standardized Periods'
      },
      'Management': {
        framework: 'Management Reporting',
        measurement_focus: 'Business Performance',
        currency_requirements: 'Constant Currency Optional',
        period_requirements: 'Business Relevant'
      }
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

  // Clean and standardize measurement terms
  cleanMeasureTerm(term) {
    try {
      if (!term) return '';
      
      let cleaned = String(term).replace(/\s+/g, ' ').trim();

    // Expand abbreviations
    Object.entries(this.abbreviationMappings).forEach(([abbr, expansion]) => {
      const regex = new RegExp(`\\b${abbr}\\b`, 'gi');
      cleaned = cleaned.replace(regex, expansion);
    });

    // Standardize common patterns
    cleaned = cleaned
      .replace(/\s*-\s*/g, ' - ')
      .replace(/\s*&\s*/g, ' and ')
      .replace(/\s*\/\s*/g, ' / ')
      .replace(/\bfx\b/gi, 'Foreign Exchange')
      .replace(/\bccy\b/gi, 'Currency')
      .replace(/\bcur\b/gi, 'Currency');

    // Title case with financial exceptions
    cleaned = cleaned.replace(/\b\w+/g, (word) => {
      const lowercaseWords = ['and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'the', 'a', 'an'];
      const uppercaseWords = ['FX', 'USD', 'EUR', 'GBP', 'IFRS', 'GAAP', 'MTD', 'YTD', 'QTD'];
      
      if (lowercaseWords.includes(word.toLowerCase())) {
        return word.toLowerCase();
      } else if (uppercaseWords.includes(word.toUpperCase())) {
        return word.toUpperCase();
      } else {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
    });

    return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
    } catch (error) {
      this.logger.error(`Error cleaning measure term: ${error.message}`);
      return term || '';
    }
  }

  // Classify measurement configuration
  classifyMeasurement(version, measureType, currencyType) {
    try {
      const versionStd = this.cleanMeasureTerm(version);
      const measureStd = this.cleanMeasureTerm(measureType);
      const currencyStd = this.cleanMeasureTerm(currencyType);

    // Get classifications
    const versionClass = this.versionClassifications[versionStd] || {
      period_type: 'Unknown',
      granularity: 'Unknown',
      reset_frequency: 'Unknown',
      typical_usage: 'Unknown',
      comparison_basis: 'Unknown'
    };

    const measureClass = this.measurementClassifications[measureStd] || {
      category: 'Unknown',
      subcategory: 'Unknown',
      data_source: 'Unknown',
      timing: 'Unknown',
      accuracy: 'Unknown',
      regulatory_usage: 'Unknown',
      audit_requirement: 'Unknown'
    };

    const currencyClass = this.currencyClassifications[currencyStd] || {
      fx_treatment: 'Unknown',
      purpose: 'Unknown',
      volatility_impact: 'Unknown',
      usage: 'Unknown',
      calculation_method: 'Unknown',
      regulatory_compliance: 'Unknown'
    };

    return {
      version: versionClass,
      measure: measureClass,
      currency: currencyClass
    };
    } catch (error) {
      this.logger.error(`Error classifying measurement: ${error.message}`);
      return {
        version: { period_type: 'Unknown', granularity: 'Unknown', reset_frequency: 'Unknown', typical_usage: 'Unknown', comparison_basis: 'Unknown' },
        measure: { category: 'Unknown', subcategory: 'Unknown', data_source: 'Unknown', timing: 'Unknown', accuracy: 'Unknown', regulatory_usage: 'Unknown', audit_requirement: 'Unknown' },
        currency: { fx_treatment: 'Unknown', purpose: 'Unknown', volatility_impact: 'Unknown', usage: 'Unknown', calculation_method: 'Unknown', regulatory_compliance: 'Unknown' }
      };
    }
  }

  // Determine applicable reporting frameworks
  determineReportingFrameworks(measureType, currencyType) {
    try {
      const frameworks = [];
    
    if (measureType.toLowerCase().includes('actual')) {
      frameworks.push('IFRS', 'GAAP');
    }
    
    if (currencyType.toLowerCase().includes('reporting')) {
      frameworks.push('IFRS', 'GAAP');
    }
    
    if (currencyType.toLowerCase().includes('constant')) {
      frameworks.push('Management');
    }
    
    if (measureType.toLowerCase().includes('budget') || measureType.toLowerCase().includes('forecast')) {
      frameworks.push('Management');
    }

    return frameworks.length > 0 ? frameworks : ['Management'];
    } catch (error) {
      this.logger.error(`Error determining reporting frameworks: ${error.message}`);
      return ['Management'];
    }
  }

  // Generate measure codes
  generateMeasureCode(version, year, measureType, currencyType, index) {
    try {
      const versionCode = version.substring(0, 2).toUpperCase(); // MT for MTD
      const yearCode = String(year).slice(-2); // Last 2 digits of year
      const measureCode = measureType.substring(0, 1).toUpperCase(); // A for Act, B for Bud
      const currencyCode = currencyType.substring(0, 2).toUpperCase(); // CF for CFX, etc.
      
      return `${versionCode}${yearCode}${measureCode}${currencyCode}`;
    } catch (error) {
      this.logger.error(`Error generating measure code: ${error.message}`);
      return `UNK${String(index).padStart(4, '0')}`;
    }
  }

  // Validate measurement configuration
  validateMeasureConfig(record) {
    try {
      const issues = [];
      
      if (!this.config.enableValidation) {
        return issues;
      }
    
    // Year validation
    const currentYear = new Date().getFullYear();
    if (record.forYear < 2020 || record.forYear > currentYear + 5) {
      issues.push(`Year ${record.forYear} seems outside reasonable range`);
    }
    
    // Logical consistency checks
    if (record.measureType === 'Bud' && record.forYear < currentYear) {
      issues.push('Budget data for historical years may indicate data quality issue');
    }
    
    // Currency type combinations
    const validCombinations = [
      ['CFX', 'RFX'], // Common combination
      ['CFX', 'FWD'], // With forward rates
      ['RFX', 'FWD']  // Reporting with forward
    ];
    
    return issues;
    } catch (error) {
      this.logger.error(`Error validating measure config: ${error.message}`);
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
        original_version: row.Version,
        original_year: row.forYear,
        original_measure_type: row.measureType,
        original_currency_type: row.currencyType
      };

      // Clean and standardize terms
      result.version_standardized = this.cleanMeasureTerm(row.Version);
      result.measure_type_standardized = this.cleanMeasureTerm(row.measureType);
      result.currency_type_standardized = this.cleanMeasureTerm(row.currencyType);
      result.reporting_year = row.forYear;

      // Classification
      const classification = this.classifyMeasurement(
        row.Version, row.measureType, row.currencyType
      );
      
      // Version classification
      result.period_type = classification.version.period_type;
      result.granularity = classification.version.granularity;
      result.reset_frequency = classification.version.reset_frequency;
      result.typical_usage = classification.version.typical_usage;
      result.comparison_basis = classification.version.comparison_basis;

      // Measure classification
      result.measure_category = classification.measure.category;
      result.measure_subcategory = classification.measure.subcategory;
      result.data_source = classification.measure.data_source;
      result.timing = classification.measure.timing;
      result.accuracy = classification.measure.accuracy;
      result.regulatory_usage = classification.measure.regulatory_usage;
      result.audit_requirement = classification.measure.audit_requirement;

      // Currency classification
      result.fx_treatment = classification.currency.fx_treatment;
      result.currency_purpose = classification.currency.purpose;
      result.volatility_impact = classification.currency.volatility_impact;
      result.currency_usage = classification.currency.usage;
      result.calculation_method = classification.currency.calculation_method;
      result.regulatory_compliance = classification.currency.regulatory_compliance;

      // Reporting frameworks
      result.applicable_frameworks = this.determineReportingFrameworks(
        row.measureType, row.currencyType
      ).join(', ');

      // Generate measure code
      result.generated_measure_code = this.generateMeasureCode(
        row.Version, row.forYear, row.measureType, row.currencyType, index
      );

      // Validation
      const validationIssues = this.validateMeasureConfig(row);
      result.validation_issues = validationIssues.join('; ') || null;

      // Quality assessment
      result.standardization_quality = this.assessStandardizationQuality(result);

        results.push(result);
        this.stats.successfulStandardizations++;
      } catch (error) {
        this.logger.error(`Error processing row ${row._row_number}: ${error.message}`);
        this.stats.errors++;
        
        // Add error record
        results.push({
          original_row_number: row._row_number,
          original_version: row.Version,
          original_year: row.forYear,
          original_measure_type: row.measureType,
          original_currency_type: row.currencyType,
          error: error.message,
          standardization_quality: 'Error'
        });
      }
    }
    
    return results;
  }

  // Process entire dataset with batch processing
  async standardizeDataset(data) {
    this.logger.info(`Starting measure standardization for ${data.length} records`);
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
    
    this.logger.info(`Measure standardization completed in ${duration.toFixed(2)} seconds`);
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

    // Standardization improvement (1 point)
    if (record.version_standardized !== record.original_version ||
        record.measure_type_standardized !== record.original_measure_type ||
        record.currency_type_standardized !== record.original_currency_type) {
      score += 1;
    }

    // Classification completeness (1 point)
    if (record.measure_category !== 'Unknown') score += 1;

    // Currency treatment identification (1 point)
    if (record.fx_treatment !== 'Unknown') score += 1;

    // Framework applicability (1 point)
    if (record.applicable_frameworks && record.applicable_frameworks !== 'Unknown') score += 1;

    // Validation passed (1 point)
    if (!record.validation_issues) score += 1;

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
      standardizationStats: {
        yearsSpanned: [...new Set(standardizedData.map(r => r.reporting_year))].length,
        measureTypes: [...new Set(standardizedData.map(r => r.measure_type_standardized))].length,
        currencyTypes: [...new Set(standardizedData.map(r => r.currency_type_standardized))].length,
        versionsFound: [...new Set(standardizedData.map(r => r.version_standardized))].length
      },
      qualityBreakdown: {
        excellent: standardizedData.filter(r => r.standardization_quality === 'Excellent').length,
        good: standardizedData.filter(r => r.standardization_quality === 'Good').length,
        fair: standardizedData.filter(r => r.standardization_quality === 'Fair').length,
        needsReview: standardizedData.filter(r => r.standardization_quality === 'Needs Review').length
      },
      measureTypeBreakdown: {},
      currencyTypeBreakdown: {},
      recommendations: []
    };

    // Breakdowns
    standardizedData.forEach(record => {
      const measureType = record.measure_category;
      const currencyType = record.fx_treatment;
      
      report.measureTypeBreakdown[measureType] = (report.measureTypeBreakdown[measureType] || 0) + 1;
      report.currencyTypeBreakdown[currencyType] = (report.currencyTypeBreakdown[currencyType] || 0) + 1;
    });

    // Generate recommendations
    const currentYear = new Date().getFullYear();
    const futureYears = standardizedData.filter(r => r.reporting_year > currentYear).length;
    if (futureYears > 0) {
      report.recommendations.push(`${futureYears} configurations for future years - ensure budget/forecast data availability`);
    }

    const uniqueFrameworks = [...new Set(standardizedData.map(r => r.applicable_frameworks))];
    if (uniqueFrameworks.length > 1) {
      report.recommendations.push('Multiple reporting frameworks detected - ensure consistency across frameworks');
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
  module.exports = MeasureStandardizer;
} else {
  window.MeasureStandardizer = MeasureStandardizer;
}