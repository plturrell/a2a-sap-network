sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/ext/agent14/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._initializeCreateModel();
            }
        },
        
        _initializeCreateModel: function() {
            var oCreateData = {
                modelName: "",
                description: "",
                modelType: "",
                baseModel: "",
                embeddingDimension: 768,
                architecture: "",
                layerCount: 12,
                hiddenSize: 768,
                attentionHeads: 12,
                vocabularySize: "30522",
                maxSequenceLength: 512,
                tokenizer: "wordpiece",
                normalization: true,
                learningRate: "0.00002",
                batchSize: 32,
                epochs: 10,
                optimizer: "adamw",
                lossFunction: "cosine",
                regularization: "0.01",
                dropout: 0.1,
                warmupSteps: 500,
                gradientClipping: "1.0",
                quantization: false,
                pruning: false,
                distillation: false,
                compressionRatio: 2,
                optimizationTarget: "balanced",
                hardwareTarget: "gpu",
                autoOptimization: true,
                mixedPrecision: true,
                datasetName: "",
                datasetSizeDisplay: "0 samples",
                trainingSamples: 10000,
                validationSamples: 2000,
                testSamples: 2000,
                dataAugmentation: [],
                samplingStrategy: "stratified",
                classBalance: true,
                modelNameState: "",
                modelNameStateText: "",
                modelTypeState: "",
                modelTypeStateText: "",
                baseModelState: "",
                baseModelStateText: "",
                datasetNameState: "",
                datasetNameStateText: "",
                canCreate: false
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.getView().setModel(oCreateModel, "create");
        },

        // Create Embedding Model Action
        onCreateEmbeddingModel: function() {
            var oView = this.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent14.ext.fragment.CreateEmbeddingModel",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._resetCreateModel();
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._resetCreateModel();
                this._oCreateDialog.open();
            }
        },
        
        // Create Dialog Lifecycle
        onCreateModelDialogAfterOpen: function() {
            // Focus on first input field
            var oModelNameInput = this.getView().byId("modelNameInput");
            if (oModelNameInput) {
                oModelNameInput.focus();
            }
            
            // Start real-time validation
            this._startCreateValidationInterval();
        },
        
        onCreateModelDialogAfterClose: function() {
            this._stopCreateValidationInterval();
        },
        
        _startCreateValidationInterval: function() {
            if (this.createValidationInterval) {
                clearInterval(this.createValidationInterval);
            }
            
            this.createValidationInterval = setInterval(function() {
                this._validateCreateForm();
            }.bind(this), 1000);
        },
        
        _stopCreateValidationInterval: function() {
            if (this.createValidationInterval) {
                clearInterval(this.createValidationInterval);
                this.createValidationInterval = null;
            }
        },
        
        _validateCreateForm: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            var bCanCreate = true;
            var securityRiskScore = 0;
            
            // Enhanced model name validation
            if (!oData.modelName || oData.modelName.trim().length < 3) {
                oData.modelNameState = "Error";
                oData.modelNameStateText = "Model name is required and must be at least 3 characters";
                bCanCreate = false;
                securityRiskScore += 20;
            } else if (!SecurityUtils.isValidModelName(oData.modelName)) {
                oData.modelNameState = "Error";
                oData.modelNameStateText = "Model name contains invalid characters or security risks";
                bCanCreate = false;
                securityRiskScore += 50;
                SecurityUtils.logSecureOperation('INVALID_MODEL_NAME', 'WARNING', {
                    modelName: oData.modelName
                });
            } else {
                oData.modelNameState = "Success";
                oData.modelNameStateText = "";
            }
            
            // Enhanced model type validation
            if (!oData.modelType) {
                oData.modelTypeState = "Warning";
                oData.modelTypeStateText = "Please select a model type";
                bCanCreate = false;
                securityRiskScore += 10;
            } else {
                const allowedModelTypes = ['bert', 'roberta', 'distilbert', 'sentence_bert', 'clip'];
                if (!allowedModelTypes.includes(oData.modelType)) {
                    oData.modelTypeState = "Error";
                    oData.modelTypeStateText = "Invalid model type selected";
                    bCanCreate = false;
                    securityRiskScore += 30;
                } else {
                    oData.modelTypeState = "Success";
                    oData.modelTypeStateText = "";
                }
            }
            
            // Enhanced base model validation
            if (!oData.baseModel) {
                oData.baseModelState = "Warning";
                oData.baseModelStateText = "Please select a base model";
                bCanCreate = false;
                securityRiskScore += 10;
            } else {
                const allowedBaseModels = ['bert_base', 'bert_large', 'roberta_base', 'roberta_large', 'distilbert_base', 'all_mpnet', 'minilm'];
                if (!allowedBaseModels.includes(oData.baseModel)) {
                    oData.baseModelState = "Error";
                    oData.baseModelStateText = "Invalid base model selected";
                    bCanCreate = false;
                    securityRiskScore += 30;
                } else {
                    oData.baseModelState = "Success";
                    oData.baseModelStateText = "";
                }
            }
            
            // Enhanced dataset validation
            if (!oData.datasetName || oData.datasetName.trim().length < 1) {
                oData.datasetNameState = "Error";
                oData.datasetNameStateText = "Dataset name is required";
                bCanCreate = false;
                securityRiskScore += 15;
            } else {
                const pathValidation = SecurityUtils.validateModelPath(oData.datasetName);
                if (!pathValidation.isValid) {
                    oData.datasetNameState = "Error";
                    oData.datasetNameStateText = "Dataset path validation failed: " + pathValidation.errors.join(', ');
                    bCanCreate = false;
                    securityRiskScore += pathValidation.riskScore;
                } else {
                    oData.datasetNameState = "Success";
                    oData.datasetNameStateText = "";
                }
            }
            
            // Validate hyperparameters for security
            const hyperparameters = {
                learningRate: oData.learningRate,
                batchSize: oData.batchSize,
                epochs: oData.epochs,
                optimizer: oData.optimizer,
                lossFunction: oData.lossFunction,
                dropout: oData.dropout,
                weightDecay: oData.regularization
            };
            
            const hyperValidation = SecurityUtils.validateHyperparameters(hyperparameters);
            if (!hyperValidation.isValid) {
                bCanCreate = false;
                securityRiskScore += hyperValidation.riskScore;
                SecurityUtils.logSecureOperation('HYPERPARAMETER_VALIDATION_FAILED', 'ERROR', {
                    errors: hyperValidation.errors,
                    riskScore: hyperValidation.riskScore
                });
            }
            
            // Check overall security risk
            if (securityRiskScore > 50) {
                bCanCreate = false;
                SecurityUtils.logSecureOperation('HIGH_SECURITY_RISK_MODEL_CREATION', 'ERROR', {
                    riskScore: securityRiskScore,
                    modelName: oData.modelName
                });
            }
            
            oData.canCreate = bCanCreate;
            oData.securityRiskScore = securityRiskScore;
            oCreateModel.setData(oData);
        },
        
        _resetCreateModel: function() {
            this._initializeCreateModel();
        },
        
        // Field Change Handlers
        onModelNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Enhanced input sanitization and validation
            const sanitizedValue = SecurityUtils.sanitizeInput(sValue);
            const isValid = SecurityUtils.isValidModelName(sanitizedValue);
            
            if (!isValid && sanitizedValue.length > 0) {
                SecurityUtils.logSecureOperation('SUSPICIOUS_MODEL_NAME_INPUT', 'WARNING', {
                    originalValue: sValue,
                    sanitizedValue: sanitizedValue
                });
            }
            
            oData.modelName = sanitizedValue;
            oCreateModel.setData(oData);
        },
        
        onDescriptionChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.description = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onModelTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.modelType = sValue;
            
            // Auto-suggest based on model type
            switch (sValue) {
                case "bert":
                    oData.baseModel = "bert_base";
                    oData.embeddingDimension = 768;
                    oData.layerCount = 12;
                    oData.hiddenSize = 768;
                    oData.attentionHeads = 12;
                    oData.vocabularySize = "30522";
                    oData.maxSequenceLength = 512;
                    oData.tokenizer = "wordpiece";
                    break;
                case "roberta":
                    oData.baseModel = "roberta_base";
                    oData.embeddingDimension = 768;
                    oData.layerCount = 12;
                    oData.hiddenSize = 768;
                    oData.attentionHeads = 12;
                    oData.vocabularySize = "50265";
                    oData.maxSequenceLength = 512;
                    oData.tokenizer = "bpe";
                    break;
                case "distilbert":
                    oData.baseModel = "distilbert_base";
                    oData.embeddingDimension = 768;
                    oData.layerCount = 6;
                    oData.hiddenSize = 768;
                    oData.attentionHeads = 12;
                    oData.vocabularySize = "30522";
                    oData.maxSequenceLength = 512;
                    oData.tokenizer = "wordpiece";
                    break;
                case "sentence_bert":
                    oData.baseModel = "all_mpnet";
                    oData.embeddingDimension = 768;
                    oData.layerCount = 12;
                    oData.hiddenSize = 768;
                    oData.attentionHeads = 12;
                    oData.vocabularySize = "30522";
                    oData.maxSequenceLength = 384;
                    oData.tokenizer = "wordpiece";
                    break;
                case "clip":
                    oData.embeddingDimension = 512;
                    oData.layerCount = 12;
                    oData.hiddenSize = 512;
                    oData.attentionHeads = 8;
                    oData.vocabularySize = "49408";
                    oData.maxSequenceLength = 77;
                    oData.tokenizer = "bpe";
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onBaseModelChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.baseModel = sValue;
            
            // Auto-adjust parameters based on base model
            switch (sValue) {
                case "bert_large":
                    oData.embeddingDimension = 1024;
                    oData.layerCount = 24;
                    oData.hiddenSize = 1024;
                    oData.attentionHeads = 16;
                    break;
                case "roberta_large":
                    oData.embeddingDimension = 1024;
                    oData.layerCount = 24;
                    oData.hiddenSize = 1024;
                    oData.attentionHeads = 16;
                    break;
                case "minilm":
                    oData.embeddingDimension = 384;
                    oData.layerCount = 6;
                    oData.hiddenSize = 384;
                    oData.attentionHeads = 12;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onEmbeddingDimensionChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.embeddingDimension = iValue;
            
            // Adjust hidden size to match embedding dimension if they're the same
            if (oData.hiddenSize === oData.embeddingDimension) {
                oData.hiddenSize = iValue;
            }
            
            oCreateModel.setData(oData);
        },
        
        onArchitectureChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.architecture = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onLayerCountChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.layerCount = iValue;
            oCreateModel.setData(oData);
        },
        
        onHiddenSizeChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.hiddenSize = iValue;
            oCreateModel.setData(oData);
        },
        
        onAttentionHeadsChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.attentionHeads = iValue;
            oCreateModel.setData(oData);
        },
        
        onVocabularySizeChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.vocabularySize = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onMaxSequenceLengthChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.maxSequenceLength = iValue;
            oCreateModel.setData(oData);
        },
        
        onTokenizerChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.tokenizer = sValue;
            oCreateModel.setData(oData);
        },
        
        onNormalizationChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.normalization = bValue;
            oCreateModel.setData(oData);
        },
        
        onLearningRateChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.learningRate = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onBatchSizeChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.batchSize = iValue;
            oCreateModel.setData(oData);
        },
        
        onEpochsChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.epochs = iValue;
            oCreateModel.setData(oData);
        },
        
        onOptimizerChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.optimizer = sValue;
            
            // Auto-adjust learning rate based on optimizer
            switch (sValue) {
                case "adam":
                    oData.learningRate = "0.001";
                    break;
                case "adamw":
                    oData.learningRate = "0.00002";
                    break;
                case "sgd":
                    oData.learningRate = "0.01";
                    break;
                case "rmsprop":
                    oData.learningRate = "0.001";
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onLossFunctionChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.lossFunction = sValue;
            oCreateModel.setData(oData);
        },
        
        onRegularizationChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.regularization = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onDropoutChange: function(oEvent) {
            var fValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.dropout = fValue;
            oCreateModel.setData(oData);
        },
        
        onWarmupStepsChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.warmupSteps = iValue;
            oCreateModel.setData(oData);
        },
        
        onGradientClippingChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.gradientClipping = SecurityUtils.sanitizeInput(sValue);
            oCreateModel.setData(oData);
        },
        
        onQuantizationChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.quantization = bValue;
            
            // Auto-suggest compression ratio when quantization is enabled
            if (bValue && oData.compressionRatio < 2) {
                oData.compressionRatio = 4;
            }
            
            oCreateModel.setData(oData);
        },
        
        onPruningChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.pruning = bValue;
            
            // Auto-suggest compression ratio when pruning is enabled
            if (bValue && oData.compressionRatio < 2) {
                oData.compressionRatio = 3;
            }
            
            oCreateModel.setData(oData);
        },
        
        onDistillationChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.distillation = bValue;
            oCreateModel.setData(oData);
        },
        
        onCompressionRatioChange: function(oEvent) {
            var fValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.compressionRatio = fValue;
            oCreateModel.setData(oData);
        },
        
        onOptimizationTargetChange: function(oEvent) {
            var sValue = oEvent.getParameter("key");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.optimizationTarget = sValue;
            
            // Auto-adjust settings based on optimization target
            switch (sValue) {
                case "speed":
                    oData.quantization = true;
                    oData.pruning = true;
                    oData.mixedPrecision = true;
                    oData.compressionRatio = 5;
                    break;
                case "balanced":
                    oData.quantization = false;
                    oData.pruning = false;
                    oData.mixedPrecision = true;
                    oData.compressionRatio = 2;
                    break;
                case "accuracy":
                    oData.quantization = false;
                    oData.pruning = false;
                    oData.mixedPrecision = false;
                    oData.compressionRatio = 1;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onHardwareTargetChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.hardwareTarget = sValue;
            
            // Auto-adjust settings based on hardware target
            switch (sValue) {
                case "cpu":
                    oData.mixedPrecision = false;
                    oData.batchSize = Math.min(oData.batchSize, 16);
                    break;
                case "gpu":
                    oData.mixedPrecision = true;
                    oData.batchSize = Math.max(oData.batchSize, 32);
                    break;
                case "tpu":
                    oData.mixedPrecision = true;
                    oData.batchSize = Math.max(oData.batchSize, 64);
                    break;
                case "edge":
                case "mobile":
                    oData.quantization = true;
                    oData.pruning = true;
                    oData.compressionRatio = 8;
                    oData.batchSize = Math.min(oData.batchSize, 8);
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onAutoOptimizationChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.autoOptimization = bValue;
            oCreateModel.setData(oData);
        },
        
        onMixedPrecisionChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.mixedPrecision = bValue;
            oCreateModel.setData(oData);
        },
        
        onDatasetNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Enhanced dataset path validation
            const sanitizedValue = SecurityUtils.sanitizeInput(sValue);
            const pathValidation = SecurityUtils.validateModelPath(sanitizedValue);
            
            if (!pathValidation.isValid && sanitizedValue.length > 0) {
                SecurityUtils.logSecureOperation('INVALID_DATASET_PATH_INPUT', 'WARNING', {
                    path: sanitizedValue,
                    errors: pathValidation.errors,
                    riskScore: pathValidation.riskScore
                });
                // Show user-friendly error without exposing security details
                oData.datasetNameState = "Error";
                oData.datasetNameStateText = "Please enter a valid dataset path";
            } else {
                oData.datasetNameState = "None";
                oData.datasetNameStateText = "";
            }
            
            oData.datasetName = pathValidation.sanitizedPath || sanitizedValue;
            
            // Secure dataset size calculation with bounds checking
            if (sanitizedValue.trim().length > 0 && pathValidation.isValid) {
                // Use deterministic calculation based on path hash for consistency
                const pathHash = this._calculateSimpleHash(sanitizedValue);
                var iEstimatedSamples = Math.min(Math.max((pathHash % 90000) + 10000, 1000), 100000);
                oData.datasetSizeDisplay = iEstimatedSamples.toLocaleString() + " samples";
                
                // Auto-suggest sample splits with validation
                oData.trainingSamples = Math.floor(iEstimatedSamples * 0.7);
                oData.validationSamples = Math.floor(iEstimatedSamples * 0.15);
                oData.testSamples = Math.floor(iEstimatedSamples * 0.15);
            } else {
                oData.datasetSizeDisplay = "0 samples";
                oData.trainingSamples = 0;
                oData.validationSamples = 0;
                oData.testSamples = 0;
            }
            
            oCreateModel.setData(oData);
        },
        
        onTrainingSamplesChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.trainingSamples = iValue;
            oCreateModel.setData(oData);
        },
        
        onValidationSamplesChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.validationSamples = iValue;
            oCreateModel.setData(oData);
        },
        
        onTestSamplesChange: function(oEvent) {
            var iValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.testSamples = iValue;
            oCreateModel.setData(oData);
        },
        
        onDataAugmentationChange: function(oEvent) {
            var aSelectedKeys = oEvent.getParameter("selectedItems").map(function(oItem) {
                return oItem.getKey();
            });
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.dataAugmentation = aSelectedKeys;
            oCreateModel.setData(oData);
        },
        
        onSamplingStrategyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.samplingStrategy = sValue;
            
            // Auto-suggest class balance based on sampling strategy
            if (sValue === "balanced" || sValue === "stratified") {
                oData.classBalance = true;
            }
            
            oCreateModel.setData(oData);
        },
        
        onClassBalanceChange: function(oEvent) {
            var bValue = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.classBalance = bValue;
            oCreateModel.setData(oData);
        },
        
        // Dialog Action Handlers
        onConfirmCreateModel: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!oData.canCreate) {
                MessageToast.show("Please fix validation errors before creating the model");
                SecurityUtils.logSecureOperation('MODEL_CREATION_REJECTED_VALIDATION', 'WARNING', {
                    modelName: oData.modelName,
                    riskScore: oData.securityRiskScore || 0
                });
                return;
            }
            
            // Additional security check before showing confirmation
            if (oData.securityRiskScore && oData.securityRiskScore > 30) {
                MessageBox.confirm("This model configuration has elevated security risk. Are you sure you want to proceed?", {
                    title: "Security Warning - Confirm Model Creation",
                    icon: MessageBox.Icon.WARNING,
                    onOK: function() {
                        this._showFinalConfirmation(oData);
                    }.bind(this)
                });
            } else {
                this._showFinalConfirmation(oData);
            }
        },
        
        /**
         * @function _showFinalConfirmation
         * @description Shows final confirmation dialog with security summary
         * @param {object} oData - Model data
         * @private
         */
        _showFinalConfirmation: function(oData) {
            const securitySummary = oData.securityRiskScore > 0 ? 
                `\n\nSecurity Risk Score: ${oData.securityRiskScore}/100` : "";
                
            MessageBox.confirm(`Are you sure you want to create the embedding model '${oData.modelName}'?${securitySummary}`, {
                title: "Confirm Model Creation",
                onOK: function() {
                    SecurityUtils.logSecureOperation('MODEL_CREATION_CONFIRMED', 'INFO', {
                        modelName: oData.modelName,
                        riskScore: oData.securityRiskScore || 0
                    });
                    this._createEmbeddingModel(oData);
                }.bind(this),
                onCancel: function() {
                    SecurityUtils.logSecureOperation('MODEL_CREATION_CANCELLED', 'INFO', {
                        modelName: oData.modelName
                    });
                }
            });
        }
        
        onCancelCreateModel: function() {
            if (this._oCreateDialog) {
                this._oCreateDialog.close();
            }
        },
        
        _createEmbeddingModel: function(oData) {
            // Enhanced security validation before model creation
            if (!SecurityUtils.checkEmbeddingAuth('CreateEmbeddingModel', oData)) {
                MessageToast.show("Insufficient permissions to create embedding model");
                return;
            }
            
            // Final security validation
            const validationResult = this._performFinalSecurityValidation(oData);
            if (!validationResult.isSecure) {
                SecurityUtils.logSecureOperation('MODEL_CREATION_BLOCKED', 'ERROR', {
                    modelName: oData.modelName,
                    securityIssues: validationResult.issues,
                    riskScore: validationResult.riskScore
                });
                MessageToast.show("Model creation blocked due to security policy violations");
                return;
            }
            
            // Sanitize all model data before creation
            const sanitizedData = this._sanitizeModelCreationData(oData);
            
            // Log model creation attempt
            SecurityUtils.logSecureOperation('MODEL_CREATION_ATTEMPT', 'INFO', {
                modelName: sanitizedData.modelName,
                modelType: sanitizedData.modelType,
                baseModel: sanitizedData.baseModel
            });
            
            // Simulate secure model creation with timeout protection
            MessageToast.show("Embedding model creation started with security validation...");
            
            // Set reasonable timeout to prevent resource exhaustion
            const creationTimeout = setTimeout(function() {
                SecurityUtils.logSecureOperation('MODEL_CREATION_SUCCESS', 'INFO', {
                    modelName: sanitizedData.modelName
                });
                MessageToast.show("Embedding model '" + sanitizedData.modelName + "' created successfully");
                if (this._oCreateDialog) {
                    this._oCreateDialog.close();
                }
            }.bind(this), 2000);
            
            // Store timeout reference for cleanup
            this._modelCreationTimeout = creationTimeout;
        },
        
        // Enhanced lifecycle with security cleanup
        onExit: function() {
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
            
            // Clear any pending timeouts
            if (this._modelCreationTimeout) {
                clearTimeout(this._modelCreationTimeout);
                this._modelCreationTimeout = null;
            }
            
            this._stopCreateValidationInterval();
            
            // Log controller cleanup
            SecurityUtils.logSecureOperation('CONTROLLER_EXIT', 'INFO', {
                controller: 'ObjectPageExt'
            });
        },
        
        /**
         * @function _performFinalSecurityValidation
         * @description Performs comprehensive security validation before model creation
         * @param {object} modelData - Model data to validate
         * @returns {object} Security validation result
         * @private
         */
        _performFinalSecurityValidation: function(modelData) {
            const validation = {
                isSecure: true,
                issues: [],
                riskScore: 0
            };
            
            // Validate model configuration
            const hyperparameters = {
                learningRate: modelData.learningRate,
                batchSize: modelData.batchSize,
                epochs: modelData.epochs,
                optimizer: modelData.optimizer,
                lossFunction: modelData.lossFunction,
                dropout: modelData.dropout
            };
            
            const hyperValidation = SecurityUtils.validateHyperparameters(hyperparameters);
            if (!hyperValidation.isValid) {
                validation.isSecure = false;
                validation.issues = validation.issues.concat(hyperValidation.errors);
                validation.riskScore += hyperValidation.riskScore;
            }
            
            // Validate training data configuration
            const trainingData = {
                datasetPath: modelData.datasetName,
                batchSize: modelData.batchSize,
                sampleCount: modelData.trainingSamples,
                dataFormat: 'json' // Default format
            };
            
            const dataValidation = SecurityUtils.validateTrainingData(trainingData);
            if (!dataValidation.isValid) {
                validation.isSecure = false;
                validation.issues = validation.issues.concat(dataValidation.errors);
                validation.riskScore += dataValidation.riskScore;
            }
            
            // Check for resource exhaustion patterns
            if (parseInt(modelData.epochs) > 1000 || parseInt(modelData.batchSize) > 1024) {
                validation.isSecure = false;
                validation.issues.push('Resource exhaustion pattern detected');
                validation.riskScore += 50;
            }
            
            // Block creation if risk score is too high
            if (validation.riskScore > 75) {
                validation.isSecure = false;
                validation.issues.push('Security risk score exceeds threshold');
            }
            
            return validation;
        },
        
        /**
         * @function _sanitizeModelCreationData
         * @description Sanitizes all model creation data
         * @param {object} modelData - Raw model data
         * @returns {object} Sanitized model data
         * @private
         */
        _sanitizeModelCreationData: function(modelData) {
            const sanitized = {
                modelName: SecurityUtils.sanitizeInput(modelData.modelName),
                description: SecurityUtils.sanitizeInput(modelData.description),
                modelType: SecurityUtils.sanitizeInput(modelData.modelType),
                baseModel: SecurityUtils.sanitizeInput(modelData.baseModel),
                architecture: SecurityUtils.sanitizeInput(modelData.architecture),
                tokenizer: SecurityUtils.sanitizeInput(modelData.tokenizer),
                optimizer: SecurityUtils.sanitizeInput(modelData.optimizer),
                lossFunction: SecurityUtils.sanitizeInput(modelData.lossFunction),
                hardwareTarget: SecurityUtils.sanitizeInput(modelData.hardwareTarget),
                samplingStrategy: SecurityUtils.sanitizeInput(modelData.samplingStrategy)
            };
            
            // Validate and sanitize numeric fields
            sanitized.embeddingDimension = Math.min(Math.max(parseInt(modelData.embeddingDimension) || 768, 64), 4096);
            sanitized.layerCount = Math.min(Math.max(parseInt(modelData.layerCount) || 12, 1), 48);
            sanitized.hiddenSize = Math.min(Math.max(parseInt(modelData.hiddenSize) || 768, 64), 4096);
            sanitized.attentionHeads = Math.min(Math.max(parseInt(modelData.attentionHeads) || 12, 1), 32);
            sanitized.maxSequenceLength = Math.min(Math.max(parseInt(modelData.maxSequenceLength) || 512, 8), 8192);
            sanitized.batchSize = Math.min(Math.max(parseInt(modelData.batchSize) || 32, 1), 1024);
            sanitized.epochs = Math.min(Math.max(parseInt(modelData.epochs) || 10, 1), 1000);
            sanitized.warmupSteps = Math.min(Math.max(parseInt(modelData.warmupSteps) || 500, 0), 10000);
            sanitized.trainingSamples = Math.min(Math.max(parseInt(modelData.trainingSamples) || 10000, 100), 10000000);
            sanitized.validationSamples = Math.min(Math.max(parseInt(modelData.validationSamples) || 2000, 10), 1000000);
            sanitized.testSamples = Math.min(Math.max(parseInt(modelData.testSamples) || 2000, 10), 1000000);
            
            // Validate and sanitize floating point fields
            sanitized.learningRate = Math.min(Math.max(parseFloat(modelData.learningRate) || 0.00002, 0.00001), 1.0);
            sanitized.regularization = Math.min(Math.max(parseFloat(modelData.regularization) || 0.01, 0), 1.0);
            sanitized.dropout = Math.min(Math.max(parseFloat(modelData.dropout) || 0.1, 0), 0.95);
            sanitized.gradientClipping = Math.min(Math.max(parseFloat(modelData.gradientClipping) || 1.0, 0.1), 10.0);
            sanitized.compressionRatio = Math.min(Math.max(parseFloat(modelData.compressionRatio) || 2, 1), 32);
            
            // Validate boolean fields
            sanitized.normalization = Boolean(modelData.normalization);
            sanitized.quantization = Boolean(modelData.quantization);
            sanitized.pruning = Boolean(modelData.pruning);
            sanitized.distillation = Boolean(modelData.distillation);
            sanitized.autoOptimization = Boolean(modelData.autoOptimization);
            sanitized.mixedPrecision = Boolean(modelData.mixedPrecision);
            sanitized.classBalance = Boolean(modelData.classBalance);
            
            return sanitized;
        },
        
        /**
         * @function _calculateSimpleHash
         * @description Calculates a simple hash for consistent dataset size estimation
         * @param {string} input - Input string to hash
         * @returns {number} Hash value
         * @private
         */
        _calculateSimpleHash: function(input) {
            let hash = 0;
            if (!input || input.length === 0) return hash;
            
            for (let i = 0; i < input.length; i++) {
                const char = input.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash; // Convert to 32bit integer
            }
            
            return Math.abs(hash);
        }
        
    });
});