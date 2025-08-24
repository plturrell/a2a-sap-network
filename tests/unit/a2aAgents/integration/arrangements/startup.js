sap.ui.define([
	'sap/ui/test/Opa5'
], (Opa5) => {
	'use strict';

	return Opa5.extend('com.sap.a2a.portal.test.integration.arrangements.Startup', {
		
		iStartMyApp: function (oOptions) {
			oOptions = oOptions || {};

			// Start the app with a minimal delay for better test stability
			this.iStartMyUIComponent({
				componentConfig: {
					name: 'com.sap.a2a.portal',
					async: true,
					manifest: true
				},
				hash: oOptions.hash,
				autoWait: oOptions.autoWait
			});
		},

		iStartMyAppOnADesktop: function (oOptions) {
			oOptions = oOptions || {};
			oOptions.autoWait = true;
			return this.iStartMyApp(oOptions);
		},

		iStartMyAppOnAPhone: function (oOptions) {
			oOptions = oOptions || {};
			oOptions.autoWait = true;
			
			// Configure for mobile viewport
			this.iStartMyUIComponent({
				componentConfig: {
					name: 'com.sap.a2a.portal',
					async: true,
					manifest: true
				},
				hash: oOptions.hash,
				autoWait: oOptions.autoWait,
				width: 375,
				height: 667
			});
		},

		iStartMyAppOnATablet: function (oOptions) {
			oOptions = oOptions || {};
			oOptions.autoWait = true;
			
			// Configure for tablet viewport
			this.iStartMyUIComponent({
				componentConfig: {
					name: 'com.sap.a2a.portal',
					async: true,
					manifest: true
				},
				hash: oOptions.hash,
				autoWait: oOptions.autoWait,
				width: 768,
				height: 1024
			});
		}
	});
});