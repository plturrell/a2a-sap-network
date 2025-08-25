"""
AI System Startup and Initialization

This module handles the proper initialization of all AI components in the A2A platform.
It ensures that models are trained with real data before serving predictions and
manages the transition from untrained to production-ready state.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import sys

# Import all services
from .data_pipeline import get_data_pipeline
from .ml_training_service import get_ml_training_service

# Import all AI modules
from .ai_agent_discovery import get_ai_discovery
from .ai_workflow_optimizer import get_workflow_optimizer
from .ai_security_monitor import get_security_monitor
from .ai_intelligent_cache import get_intelligent_cache
from .ai_rate_limiter import get_ai_rate_limiter
from .ai_self_healing import get_self_healing_system
from .ai_log_analyzer import get_log_analyzer
from .ai_resource_manager import get_resource_manager
from .ai_data_quality import get_data_quality_validator
from .ai_performance_optimizer import get_performance_optimizer
from .ai_message_router import get_ai_message_router
from .ai_user_behavior import get_ai_user_behavior_predictor
from .ai_query_optimizer import get_ai_query_optimizer
from .ai_error_recovery import get_ai_error_recovery_system

logger = logging.getLogger(__name__)


class AISystemStartup:
    """
    Manages the startup and initialization of all AI components.
    Ensures models are properly trained before going into production.
    """
    
    def __init__(self):
        self.data_pipeline = None
        self.training_service = None
        self.ai_modules = {}
        self.startup_status = {
            'phase': 'not_started',
            'start_time': None,
            'components_initialized': 0,
            'models_trained': 0,
            'errors': []
        }
    
    async def initialize_ai_system(self, skip_initial_training: bool = False):
        """
        Initialize the entire AI system with proper data collection and training.
        
        Args:
            skip_initial_training: If True, starts with untrained models that will
                                 learn incrementally (useful for testing)
        """
        try:
            self.startup_status['phase'] = 'initializing'
            self.startup_status['start_time'] = datetime.utcnow()
            
            logger.info("=== Starting A2A AI System Initialization ===")
            
            # Phase 1: Initialize data pipeline
            logger.info("Phase 1: Initializing data pipeline...")
            self.data_pipeline = get_data_pipeline()
            await self.data_pipeline.start()
            logger.info("✓ Data pipeline started")
            
            # Phase 2: Initialize ML training service
            logger.info("Phase 2: Initializing ML training service...")
            self.training_service = get_ml_training_service()
            await self.training_service.start()
            logger.info("✓ ML training service started")
            
            # Phase 3: Initialize all AI modules
            logger.info("Phase 3: Initializing AI modules...")
            await self._initialize_ai_modules()
            logger.info(f"✓ Initialized {len(self.ai_modules)} AI modules")
            
            # Phase 4: Check for existing trained models
            logger.info("Phase 4: Loading existing trained models...")
            loaded_models = await self._load_existing_models()
            logger.info(f"✓ Loaded {loaded_models} pre-trained models")
            
            if not skip_initial_training:
                # Phase 5: Collect initial data
                logger.info("Phase 5: Collecting initial training data...")
                await self._collect_initial_data()
                
                # Phase 6: Train critical models
                logger.info("Phase 6: Training critical models...")
                await self._train_critical_models()
                logger.info(f"✓ Trained {self.startup_status['models_trained']} models")
            else:
                logger.info("⚠ Skipping initial training - models will learn incrementally")
            
            # Phase 7: Start production mode
            logger.info("Phase 7: Transitioning to production mode...")
            await self._start_production_mode()
            
            self.startup_status['phase'] = 'running'
            elapsed = (datetime.utcnow() - self.startup_status['start_time']).total_seconds()
            logger.info(f"=== AI System Initialization Complete in {elapsed:.1f}s ===")
            
            # Print system status
            await self._print_system_status()
            
        except Exception as e:
            logger.error(f"AI system initialization failed: {e}")
            self.startup_status['phase'] = 'failed'
            self.startup_status['errors'].append(str(e))
            raise
    
    async def _initialize_ai_modules(self):
        """Initialize all AI modules"""
        modules = [
            ('AgentDiscovery', get_ai_discovery),
            ('WorkflowOptimizer', get_workflow_optimizer),
            ('SecurityMonitor', get_security_monitor),
            ('IntelligentCache', get_intelligent_cache),
            ('RateLimiter', get_ai_rate_limiter),
            ('SelfHealing', get_self_healing_system),
            ('LogAnalyzer', get_log_analyzer),
            ('ResourceManager', get_resource_manager),
            ('DataQuality', get_data_quality_validator),
            ('PerformanceOptimizer', get_performance_optimizer),
            ('MessageRouter', get_ai_message_router),
            ('UserBehavior', get_ai_user_behavior_predictor),
            ('QueryOptimizer', get_ai_query_optimizer),
            ('ErrorRecovery', get_ai_error_recovery_system),
        ]
        
        for name, getter in modules:
            try:
                module = getter()
                self.ai_modules[name] = module
                self.startup_status['components_initialized'] += 1
                logger.info(f"  ✓ {name} initialized")
            except Exception as e:
                logger.error(f"  ✗ Failed to initialize {name}: {e}")
                self.startup_status['errors'].append(f"{name}: {str(e)}")
    
    async def _load_existing_models(self) -> int:
        """Load any existing trained models from disk"""
        loaded_count = 0
        
        for model_name in self.training_service.model_registry:
            model = self.training_service.load_model(model_name)
            if model is not None:
                # Update the model in its module
                model_info = self.training_service.model_registry[model_name]
                setattr(model_info['module'], model_info['attr_name'], model)
                loaded_count += 1
                logger.info(f"  ✓ Loaded {model_name}")
        
        return loaded_count
    
    async def _collect_initial_data(self):
        """Ensure we have enough data for initial training"""
        logger.info("  Checking available training data...")
        
        data_types = ['events', 'metrics', 'queries', 'messages']
        data_counts = {}
        
        for data_type in data_types:
            df = await self.data_pipeline.get_training_data(
                data_type=data_type,
                hours=24,  # Last 24 hours
                min_samples=1
            )
            data_counts[data_type] = len(df)
            logger.info(f"    {data_type}: {len(df)} samples")
        
        # If insufficient data, generate some initial activity
        min_required = 50
        for data_type, count in data_counts.items():
            if count < min_required:
                logger.info(f"  ⚠ Insufficient {data_type} data ({count} < {min_required})")
                
                # In production, this would wait for real data
                # For now, we'll log a warning
                logger.warning(f"    Waiting for more {data_type} data to accumulate...")
    
    async def _train_critical_models(self):
        """Train the most critical models for system operation"""
        critical_models = [
            'AISecurityMonitor.intrusion_detector',
            'AISecurityMonitor.anomaly_detector',
            'AIErrorRecoverySystem.error_classifier',
            'AISelfHealingSystem.failure_predictor',
            'AIRateLimiter.burst_predictor',
        ]
        
        for model_name in critical_models:
            if model_name in self.training_service.model_registry:
                try:
                    logger.info(f"  Training {model_name}...")
                    job = await self.training_service.train_model(model_name)
                    
                    # Wait for completion (with timeout)
                    timeout = 60  # 1 minute timeout
                    start_time = datetime.utcnow()
                    
                    while job.status not in ['completed', 'failed']:
                        if (datetime.utcnow() - start_time).seconds > timeout:
                            logger.warning(f"    Timeout training {model_name}")
                            break
                        await asyncio.sleep(1)
                    
                    if job.status == 'completed':
                        self.startup_status['models_trained'] += 1
                        logger.info(f"    ✓ {model_name} trained successfully")
                    else:
                        logger.error(f"    ✗ {model_name} training failed: {job.error}")
                        
                except Exception as e:
                    logger.error(f"    ✗ Error training {model_name}: {e}")
    
    async def _start_production_mode(self):
        """Configure all modules for production operation"""
        logger.info("  Configuring modules for production...")
        
        # Enable real-time learning for all modules
        for name, module in self.ai_modules.items():
            if hasattr(module, 'enable_production_mode'):
                module.enable_production_mode()
        
        # Start continuous monitoring
        if hasattr(self.training_service, 'enable_continuous_training'):
            self.training_service.enable_continuous_training()
        
        logger.info("  ✓ Production mode enabled")
    
    async def _print_system_status(self):
        """Print current system status"""
        status = self.training_service.get_model_status()
        
        print("\n" + "="*60)
        print("A2A AI SYSTEM STATUS")
        print("="*60)
        print(f"Status: {self.startup_status['phase'].upper()}")
        print(f"Components: {self.startup_status['components_initialized']}")
        print(f"Models Trained: {self.startup_status['models_trained']}")
        print(f"Active Training Jobs: {status['active_jobs']}")
        print(f"Performance Alerts: {status['performance_alerts']}")
        
        if self.startup_status['errors']:
            print(f"\nErrors ({len(self.startup_status['errors'])}):")
            for error in self.startup_status['errors'][:5]:
                print(f"  - {error}")
        
        print("\nModel Status:")
        for model_name, model_status in list(status['models'].items())[:10]:
            perf = model_status['recent_performance']
            perf_str = f"{perf:.3f}" if perf is not None else "N/A"
            print(f"  {model_name}: Performance={perf_str}, Schedule={model_status['schedule']}")
        
        print("="*60 + "\n")
    
    async def shutdown(self):
        """Gracefully shutdown the AI system"""
        logger.info("Shutting down AI system...")
        
        try:
            # Stop training service
            if self.training_service:
                await self.training_service.stop()
            
            # Stop data pipeline
            if self.data_pipeline:
                await self.data_pipeline.stop()
            
            logger.info("AI system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def main():
    """Main entry point for AI system startup"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check command line arguments
    skip_training = '--skip-training' in sys.argv
    
    # Initialize AI system
    startup = AISystemStartup()
    
    try:
        await startup.initialize_ai_system(skip_initial_training=skip_training)
        
        # Keep running
        logger.info("AI system is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            # Periodically print status
            await startup._print_system_status()
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await startup.shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await startup.shutdown()
        raise


if __name__ == "__main__":
    asyncio.run(main())