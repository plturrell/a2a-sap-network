const fs = require('fs');
const path = require('path');

// Fix template literals in Agent 11 SecurityUtils.js
const filePath = path.join(__dirname, '../app/a2aFiori/webapp/ext/agent11/utils/SecurityUtils.js');

console.log('Reading SecurityUtils.js...');
let content = fs.readFileSync(filePath, 'utf8');

// Pattern to match template literals with variables
const templateLiteralPattern = /`([^`]*)\$\{([^}]+)\}([^`]*)`/g;
const simpleTemplateLiteralPattern = /`([^`]*)`/g;

// Replace template literals with concatenation
let fixedContent = content;
let replacementCount = 0;

// First pass: Replace template literals with ${} expressions
fixedContent = fixedContent.replace(templateLiteralPattern, (match, before, variable, after) => {
    replacementCount++;
    // Handle the replacement based on content
    if (before && after) {
        return `"${before}" + ${variable} + "${after}"`;
    } else if (before) {
        return `"${before}" + ${variable}`;
    } else if (after) {
        return `${variable} + "${after}"`;
    } else {
        return variable;
    }
});

// Second pass: Replace simple template literals without variables
fixedContent = fixedContent.replace(simpleTemplateLiteralPattern, (match, content) => {
    // Skip if it's already been processed or contains ${
    if (content.includes('${')) {
        return match;
    }
    replacementCount++;
    return `"${content}"`;
});

// Specific fixes for known problematic patterns
const specificFixes = [
    {
        pattern: /errors\.push\("SQL injection in parameter '" \+ key \+ "': " \+ message\)/g,
        replacement: 'errors.push("SQL injection in parameter \'" + key + "\': " + message)'
    },
    {
        pattern: /warnings\.push\("Parameter '" \+ key \+ "' is unusually long \(" \+ value\.length \+ " characters\)"\)/g,
        replacement: 'warnings.push("Parameter \'" + key + "\' is unusually long (" + value.length + " characters)")'
    },
    {
        pattern: /"rateLimit_" \+ userId \+ "_" \+ operation/g,
        replacement: '"rateLimit_" + userId + "_" + operation'
    }
];

specificFixes.forEach(fix => {
    fixedContent = fixedContent.replace(fix.pattern, fix.replacement);
});

console.log(`Fixed ${replacementCount} template literals`);

// Write the fixed content back
fs.writeFileSync(filePath, fixedContent, 'utf8');
console.log('SecurityUtils.js has been updated');

// Also fix the SQLUtils.js file
const sqlUtilsPath = path.join(__dirname, '../app/a2aFiori/webapp/ext/agent11/utils/SQLUtils.js');
if (fs.existsSync(sqlUtilsPath)) {
    console.log('\nFixing SQLUtils.js...');
    let sqlContent = fs.readFileSync(sqlUtilsPath, 'utf8');
    let sqlReplacementCount = 0;
    
    // Replace template literals in SQL queries
    sqlContent = sqlContent.replace(/sql:\s*`([^`]+)`/g, (match, query) => {
        sqlReplacementCount++;
        // Replace any ${} expressions in the SQL
        const fixedQuery = query.replace(/\$\{([^}]+)\}/g, '" + $1 + "');
        return `sql: "${fixedQuery}"`;
    });
    
    // Fix specific SQL template patterns
    sqlContent = sqlContent.replace(/`SELECT\s+\$\{([^}]+)\}\s+FROM\s+\$\{([^}]+)\}`/g, 
        '"SELECT " + $1 + " FROM " + $2');
    sqlContent = sqlContent.replace(/`([^`]+)\$\{([^}]+)\}([^`]+)`/g, 
        '"$1" + $2 + "$3"');
    
    fs.writeFileSync(sqlUtilsPath, sqlContent, 'utf8');
    console.log(`Fixed ${sqlReplacementCount} template literals in SQLUtils.js`);
}

console.log('\nTemplate literal fixes completed!');