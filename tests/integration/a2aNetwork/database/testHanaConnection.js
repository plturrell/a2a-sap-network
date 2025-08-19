const hana = require('@sap/hana-client');

const connOptions = {
    serverNode: 'd93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com:443',
    uid: 'DBADMIN',
    pwd: 'Initial@1',
    encrypt: true,
    sslValidateCertificate: false
};

const connection = hana.createConnection();

console.log('Testing HANA connection...');
console.log('Connection options:', { ...connOptions, pwd: '***' });

connection.connect(connOptions, function(err) {
    if (err) {
        console.error('Connection failed:', err);
        process.exit(1);
    }
    
    console.log('Successfully connected to HANA!');
    
    // Test query
    connection.exec('SELECT CURRENT_TIMESTAMP FROM DUMMY', function(err, result) {
        if (err) {
            console.error('Query failed:', err);
        } else {
            console.log('Query successful:', result);
        }
        
        connection.disconnect();
        process.exit(0);
    });
});