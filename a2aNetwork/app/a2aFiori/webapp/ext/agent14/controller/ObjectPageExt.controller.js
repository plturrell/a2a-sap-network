sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent14/ext/utils/SecurityUtils"
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
            
            // Model name validation
            if (!oData.modelName || oData.modelName.trim().length < 3) {
                oData.modelNameState = "Error";
                oData.modelNameStateText = "Model name is required and must be at least 3 characters";
                bCanCreate = false;
            } else if (!SecurityUtils.isValidModelName(oData.modelName)) {
                oData.modelNameState = "Error";
                oData.modelNameStateText = "Model name contains invalid characters";
                bCanCreate = false;
            } else {
                oData.modelNameState = "Success";
                oData.modelNameStateText = "";
            }
            
            // Model type validation
            if (!oData.modelType) {
                oData.modelTypeState = "Warning";
                oData.modelTypeStateText = "Please select a model type";
                bCanCreate = false;
            } else {
                oData.modelTypeState = "Success";
                oData.modelTypeStateText = "";
            }
            
            // Base model validation
            if (!oData.baseModel) {
                oData.baseModelState = "Warning";
                oData.baseModelStateText = "Please select a base model";
                bCanCreate = false;
            } else {
                oData.baseModelState = "Success";
                oData.baseModelStateText = "";
            }
            
            // Dataset name validation
            if (!oData.datasetName || oData.datasetName.trim().length < 1) {
                oData.datasetNameState = "Error";
                oData.datasetNameStateText = "Dataset name is required";
                bCanCreate = false;
            } else if (!SecurityUtils.isValidDatasetPath(oData.datasetName)) {
                oData.datasetNameState = "Error";
                oData.datasetNameStateText = "Dataset path contains invalid characters";
                bCanCreate = false;
            } else {
                oData.datasetNameState = "Success";
                oData.datasetNameStateText = "";
            }
            
            oData.canCreate = bCanCreate;
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
            
            oData.modelName = SecurityUtils.sanitizeInput(sValue);
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
            
            oData.datasetName = SecurityUtils.sanitizeInput(sValue);
            
            // Simulate dataset size calculation
            if (sValue.trim().length > 0) {
                var iEstimatedSamples = Math.floor(Math.random() * 90000) + 10000;
                oData.datasetSizeDisplay = iEstimatedSamples.toLocaleString() + " samples";
                
                // Auto-suggest sample splits
                oData.trainingSamples = Math.floor(iEstimatedSamples * 0.7);
                oData.validationSamples = Math.floor(iEstimatedSamples * 0.15);
                oData.testSamples = Math.floor(iEstimatedSamples * 0.15);
            } else {
                oData.datasetSizeDisplay = "0 samples";
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
                return;
            }
            
            MessageBox.confirm("Are you sure you want to create this embedding model?", {
                title: "Confirm Model Creation",
                onOK: function() {
                    this._createEmbeddingModel(oData);
                }.bind(this)
            });
        },
        
        onCancelCreateModel: function() {
            if (this._oCreateDialog) {
                this._oCreateDialog.close();
            }
        },
        
        _createEmbeddingModel: function(oData) {
            // Simulate model creation
            MessageToast.show("Embedding model creation started...");
            
            setTimeout(function() {
                MessageToast.show("Embedding model '" + oData.modelName + "' created successfully");
                if (this._oCreateDialog) {
                    this._oCreateDialog.close();
                }
            }.bind(this), 2000);
        },
        
        // Lifecycle
        onExit: function() {
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
            
            this._stopCreateValidationInterval();
        }
        
    });
});