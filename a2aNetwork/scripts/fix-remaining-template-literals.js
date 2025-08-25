const fs = require('fs');
const path = require('path');

// Fix remaining template literal issues in Agent 10 and 11
const files = [
    'app/a2aFiori/webapp/ext/agent10/utils/SecurityUtils.js',
    'app/a2aFiori/webapp/ext/agent11/utils/SecurityUtils.js',
    'app/a2aFiori/webapp/ext/agent11/utils/SQLUtils.js'
];

files.forEach(filePath => {
    const fullPath = path.join(__dirname, '..', filePath);
    if (!fs.existsSync(fullPath)) {
        console.log(`File not found: ${filePath}`);
        return;
    }

    console.log(`\nProcessing ${filePath}...`);
    let content = fs.readFileSync(fullPath, 'utf8');
    let replacementCount = 0;

    // Fix remaining template literals with ${} in strings
    const fixes = [
        // Fix rate limit keys
        {
            pattern: /"rateLimit_\$\{userId\}_" \+ operation/g,
            replacement: '"rateLimit_" + userId + "_" + operation'
        },
        // Fix SQL query issues
        {
            pattern: /issues\.push\("Too many joins \(\$\{totalJoins\}, limit: " \+ limits\.maxJoins \+ "\)"\)/g,
            replacement: 'issues.push("Too many joins (" + totalJoins + ", limit: " + limits.maxJoins + ")")'
        },
        {
            pattern: /issues\.push\("Too many subqueries \(\$\{subqueryCount\}, limit: " \+ limits\.maxSubqueries \+ "\)"\)/g,
            replacement: 'issues.push("Too many subqueries (" + subqueryCount + ", limit: " + limits.maxSubqueries + ")")'
        },
        {
            pattern: /issues\.push\("Too many UNION operations \(\$\{unionCount\}, limit: " \+ limits\.maxUnions \+ "\)"\)/g,
            replacement: 'issues.push("Too many UNION operations (" + unionCount + ", limit: " + limits.maxUnions + ")")'
        },
        {
            pattern: /issues\.push\("Too many tables referenced \(\$\{tableCount\}, limit: " \+ limits\.maxTables \+ "\)"\)/g,
            replacement: 'issues.push("Too many tables referenced (" + tableCount + ", limit: " + limits.maxTables + ")")'
        },
        // Fix warning logs
        {
            pattern: /Log\.warning\("Rate limit exceeded for user \$\{userId\} operation " \+ operation\)/g,
            replacement: 'Log.warning("Rate limit exceeded for user " + userId + " operation " + operation)'
        },
        // Fix any remaining ${} patterns in strings
        {
            pattern: /"\$\{([^}]+)\}"/g,
            replacement: '" + $1 + "'
        },
        {
            pattern: /" \+ ([^+]+) \+ " \+ ([^+]+) \+ "/g,
            replacement: '" + $1 + " + $2 + "'
        }
    ];

    fixes.forEach(fix => {
        const before = content.length;
        content = content.replace(fix.pattern, fix.replacement);
        if (content.length !== before) {
            replacementCount++;
            console.log(`Applied fix: ${fix.pattern.source.substring(0, 50)}...`);
        }
    });

    // Fix any remaining template literal issues
    const createSetClause = /const createSetClause = \(col\) => `\$\{col\} = \?`;/g;
    content = content.replace(createSetClause, 'const createSetClause = function(col) { return col + " = ?"; };');

    // Save the fixed content
    fs.writeFileSync(fullPath, content, 'utf8');
    console.log(`Fixed ${replacementCount} issues in ${filePath}`);
});

console.log('\nFix completed!');