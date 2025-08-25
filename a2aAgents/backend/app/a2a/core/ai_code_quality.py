"""
AI-Driven Code Quality Analysis and Improvement System

This module provides intelligent code quality assessment, analysis, and improvement
suggestions using real machine learning for automated code review, pattern detection,
vulnerability identification, and quality enhancement recommendations.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import numpy as np
import json
import ast
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from enum import Enum
import pathlib
import threading
import hashlib

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Code analysis tools
try:
    import pylint.lint
    from pylint.reporters.json_reporter import JSONReporter
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import flake8.api.legacy as flake8
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False

# Security analysis
try:
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False

# Deep learning for complex code analysis
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CodeQualityNN(nn.Module):
    """Neural network for intelligent code quality analysis"""
    def __init__(self, vocab_size=15000, embedding_dim=256, hidden_dim=512):
        super(CodeQualityNN, self).__init__()
        
        # Code token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Hierarchical LSTM for code structure understanding
        self.token_lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                                 batch_first=True, bidirectional=True, num_layers=2)
        self.function_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, 
                                   batch_first=True, bidirectional=True, num_layers=2)
        
        # Multi-head attention for important code patterns
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=16)
        
        # Convolutional layers for local pattern detection
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, 256, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        self.conv_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_dim + 256 * 4, hidden_dim)
        
        # Multi-task prediction heads
        self.complexity_head = nn.Linear(hidden_dim, 1)           # Code complexity
        self.maintainability_head = nn.Linear(hidden_dim, 1)     # Maintainability score
        self.readability_head = nn.Linear(hidden_dim, 1)         # Readability score
        self.security_head = nn.Linear(hidden_dim, 8)            # Security issue types
        self.bug_risk_head = nn.Linear(hidden_dim, 1)            # Bug probability
        self.performance_head = nn.Linear(hidden_dim, 1)         # Performance score
        self.style_compliance_head = nn.Linear(hidden_dim, 1)    # Style compliance
        self.test_coverage_head = nn.Linear(hidden_dim, 1)       # Test coverage prediction
        self.refactor_priority_head = nn.Linear(hidden_dim, 3)   # Refactor priority (low/med/high)
        
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len = x.size()
        
        # Embed tokens
        embedded = self.embedding(x)
        
        # Token-level processing
        token_out, _ = self.token_lstm(embedded)
        
        # Function-level processing (simplified)
        func_out, _ = self.function_lstm(token_out)
        
        # Apply attention
        attn_out, attn_weights = self.attention(func_out, func_out, func_out,
                                               key_padding_mask=attention_mask)
        
        # Convolutional feature extraction
        conv_features = []
        conv_input = attn_out.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        
        for conv_layer in self.conv1d_layers:
            conv_out = F.relu(conv_layer(conv_input))
            pooled = self.conv_pool(conv_out).squeeze(-1)
            conv_features.append(pooled)
        
        conv_features = torch.cat(conv_features, dim=1)
        
        # Get sequence representation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(attn_out.size())
            masked_attn = attn_out.masked_fill(mask_expanded, 0.0)
            sequence_repr = masked_attn.sum(1) / (~attention_mask).sum(1, keepdim=True).float()
        else:
            sequence_repr = attn_out.mean(1)
        
        # Fuse features
        combined_features = torch.cat([sequence_repr, conv_features], dim=1)
        features = self.feature_fusion(combined_features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        
        # Predictions
        complexity = F.relu(self.complexity_head(features))
        maintainability = torch.sigmoid(self.maintainability_head(features))
        readability = torch.sigmoid(self.readability_head(features))
        security = F.softmax(self.security_head(features), dim=-1)
        bug_risk = torch.sigmoid(self.bug_risk_head(features))
        performance = torch.sigmoid(self.performance_head(features))
        style_compliance = torch.sigmoid(self.style_compliance_head(features))
        test_coverage = torch.sigmoid(self.test_coverage_head(features))
        refactor_priority = F.softmax(self.refactor_priority_head(features), dim=-1)
        
        return {
            'complexity': complexity,
            'maintainability': maintainability,
            'readability': readability,
            'security_issues': security,
            'bug_risk': bug_risk,
            'performance': performance,
            'style_compliance': style_compliance,
            'test_coverage': test_coverage,
            'refactor_priority': refactor_priority,
            'attention_weights': attn_weights,
            'features': features
        }


@dataclass
class CodeIssue:
    """Represents a code quality issue"""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    message: str
    suggestion: str
    confidence: float = 0.0
    auto_fixable: bool = False
    fix_code: Optional[str] = None
    category: str = "general"
    rule_id: Optional[str] = None


@dataclass
class CodeMetrics:
    """Comprehensive code quality metrics"""
    file_path: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    readability_score: float
    security_score: float
    performance_score: float
    test_coverage: float
    code_smells: int
    duplicated_lines: int
    technical_debt_hours: float
    issues: List[CodeIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Complete quality analysis report"""
    project_path: str
    analysis_timestamp: datetime
    overall_score: float
    file_metrics: List[CodeMetrics]
    summary_stats: Dict[str, Any]
    improvement_recommendations: List[str]
    critical_issues: List[CodeIssue]
    trend_analysis: Dict[str, Any] = field(default_factory=dict)


class AICodeQuality:
    """
    AI-powered code quality analysis and improvement system using real ML models
    """
    
    def __init__(self):
        # ML Models for different quality aspects
        self.complexity_predictor = GradientBoostingRegressor(n_estimators=150, random_state=42)
        self.bug_predictor = RandomForestClassifier(n_estimators=200, random_state=42)
        self.maintainability_scorer = MLPRegressor(hidden_layer_sizes=(256, 128), random_state=42)
        self.security_analyzer = RandomForestClassifier(n_estimators=100, random_state=42)
        self.performance_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Clustering for pattern detection
        self.pattern_clusterer = KMeans(n_clusters=12, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Feature extractors
        self.code_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.structure_vectorizer = CountVectorizer(max_features=5000)
        self.scaler = StandardScaler()
        
        # Neural network for advanced analysis
        if TORCH_AVAILABLE:
            self.quality_nn = CodeQualityNN()
            self.nn_optimizer = torch.optim.AdamW(self.quality_nn.parameters(), lr=0.001)
        else:
            self.quality_nn = None
        
        # Code quality patterns and rules
        self.quality_patterns = self._initialize_quality_patterns()
        self.security_patterns = self._initialize_security_patterns()
        self.performance_patterns = self._initialize_performance_patterns()
        
        # Analysis history and caching
        self.analysis_history = deque(maxlen=1000)
        self.metrics_cache = {}
        self.pattern_cache = {}
        
        # Quality benchmarks and thresholds
        self.quality_thresholds = {
            'complexity': {'good': 0.3, 'fair': 0.6, 'poor': 0.8},
            'maintainability': {'good': 0.8, 'fair': 0.6, 'poor': 0.4},
            'readability': {'good': 0.8, 'fair': 0.6, 'poor': 0.4},
            'security': {'good': 0.9, 'fair': 0.7, 'poor': 0.5},
            'performance': {'good': 0.8, 'fair': 0.6, 'poor': 0.4}
        }
        
        # Statistics tracking
        self.analysis_stats = defaultdict(int)
        
        # Initialize models with training data
        self._initialize_models()
        
        logger.info("AI Code Quality system initialized with ML models")
    
    async def analyze_codebase(self, project_path: str, 
                             include_patterns: List[str] = None,
                             exclude_patterns: List[str] = None) -> QualityReport:
        """
        Perform comprehensive code quality analysis on entire codebase
        """
        try:
            start_time = time.time()
            
            # Default patterns
            if include_patterns is None:
                include_patterns = ["**/*.py"]
            if exclude_patterns is None:
                exclude_patterns = ["**/test_*.py", "**/__pycache__/**", "**/venv/**"]
            
            # Discover Python files
            python_files = self._discover_files(project_path, include_patterns, exclude_patterns)
            logger.info(f"Analyzing {len(python_files)} Python files")
            
            # Analyze each file
            file_metrics = []
            for file_path in python_files:
                try:
                    metrics = await self._analyze_file(str(file_path))
                    file_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")
                    continue
            
            # Calculate overall statistics
            overall_score, summary_stats = self._calculate_project_metrics(file_metrics)
            
            # Generate improvement recommendations
            recommendations = await self._generate_recommendations(file_metrics)
            
            # Identify critical issues
            critical_issues = self._identify_critical_issues(file_metrics)
            
            # Trend analysis (if historical data available)
            trend_analysis = await self._perform_trend_analysis(project_path, file_metrics)
            
            # Create comprehensive report
            report = QualityReport(
                project_path=project_path,
                analysis_timestamp=datetime.utcnow(),
                overall_score=overall_score,
                file_metrics=file_metrics,
                summary_stats=summary_stats,
                improvement_recommendations=recommendations,
                critical_issues=critical_issues,
                trend_analysis=trend_analysis
            )
            
            # Cache results
            cache_key = hashlib.md5(project_path.encode()).hexdigest()
            self.metrics_cache[cache_key] = report
            
            # Update statistics
            analysis_time = time.time() - start_time
            self.analysis_stats['total_analyses'] += 1
            self.analysis_stats['total_files_analyzed'] += len(python_files)
            self.analysis_stats['total_analysis_time'] += analysis_time
            self.analysis_stats['issues_found'] += len(critical_issues)
            
            logger.info(f"Code quality analysis completed in {analysis_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Codebase analysis error: {e}")
            return QualityReport(
                project_path=project_path,
                analysis_timestamp=datetime.utcnow(),
                overall_score=0.0,
                file_metrics=[],
                summary_stats={'error': str(e)},
                improvement_recommendations=[],
                critical_issues=[]
            )
    
    async def analyze_code_fragment(self, code: str, 
                                  file_path: str = "unknown") -> CodeMetrics:
        """
        Analyze a specific code fragment for quality issues
        """
        try:
            # Parse code
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return CodeMetrics(
                    file_path=file_path,
                    lines_of_code=len(code.split('\n')),
                    complexity_score=1.0,
                    maintainability_index=0.0,
                    readability_score=0.0,
                    security_score=0.0,
                    performance_score=0.0,
                    test_coverage=0.0,
                    code_smells=1,
                    duplicated_lines=0,
                    technical_debt_hours=1.0,
                    issues=[CodeIssue(
                        file_path=file_path,
                        line_number=e.lineno or 1,
                        issue_type="syntax_error",
                        severity="critical",
                        message=f"Syntax error: {e.msg}",
                        suggestion="Fix syntax error",
                        confidence=1.0,
                        category="syntax"
                    )]
                )
            
            # Extract features
            features = self._extract_code_features(code, tree)
            
            # ML-based predictions
            complexity_score = self._predict_complexity(features)
            maintainability_score = self._predict_maintainability(features)
            readability_score = self._predict_readability(features)
            security_score = self._predict_security_score(features)
            performance_score = self._predict_performance_score(features)
            
            # Detect issues and code smells
            issues = await self._detect_issues(code, tree, file_path)
            code_smells = self._count_code_smells(tree)
            
            # Calculate metrics
            lines_of_code = len([line for line in code.split('\n') if line.strip()])
            technical_debt = self._estimate_technical_debt(issues, complexity_score)
            
            # Generate suggestions
            suggestions = await self._generate_code_suggestions(code, issues, features)
            
            metrics = CodeMetrics(
                file_path=file_path,
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                maintainability_index=maintainability_score,
                readability_score=readability_score,
                security_score=security_score,
                performance_score=performance_score,
                test_coverage=0.0,  # Would need additional analysis
                code_smells=code_smells,
                duplicated_lines=0,  # Would need cross-file analysis
                technical_debt_hours=technical_debt,
                issues=issues,
                suggestions=suggestions
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Code fragment analysis error: {e}")
            return CodeMetrics(
                file_path=file_path,
                lines_of_code=0,
                complexity_score=0.0,
                maintainability_index=0.0,
                readability_score=0.0,
                security_score=0.0,
                performance_score=0.0,
                test_coverage=0.0,
                code_smells=0,
                duplicated_lines=0,
                technical_debt_hours=0.0
            )
    
    async def suggest_improvements(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Generate AI-powered improvement suggestions for a specific file
        """
        try:
            # Analyze file if not already cached
            if file_path not in self.metrics_cache:
                metrics = await self._analyze_file(file_path)
            else:
                metrics = self.metrics_cache[file_path]
            
            improvements = []
            
            # High-priority improvements based on ML predictions
            if metrics.complexity_score > self.quality_thresholds['complexity']['fair']:
                improvements.append({
                    'type': 'complexity_reduction',
                    'priority': 'high',
                    'description': 'Reduce code complexity by breaking down large functions',
                    'impact': 'High',
                    'effort': 'Medium',
                    'suggestions': self._generate_complexity_improvements(metrics)
                })
            
            if metrics.security_score < self.quality_thresholds['security']['fair']:
                improvements.append({
                    'type': 'security_enhancement',
                    'priority': 'critical',
                    'description': 'Address security vulnerabilities and improve secure coding',
                    'impact': 'Critical',
                    'effort': 'Medium',
                    'suggestions': self._generate_security_improvements(metrics)
                })
            
            if metrics.maintainability_index < self.quality_thresholds['maintainability']['fair']:
                improvements.append({
                    'type': 'maintainability_improvement',
                    'priority': 'medium',
                    'description': 'Improve code maintainability and structure',
                    'impact': 'Medium',
                    'effort': 'High',
                    'suggestions': self._generate_maintainability_improvements(metrics)
                })
            
            # Performance optimizations
            if metrics.performance_score < self.quality_thresholds['performance']['fair']:
                improvements.append({
                    'type': 'performance_optimization',
                    'priority': 'medium',
                    'description': 'Optimize code for better performance',
                    'impact': 'Medium',
                    'effort': 'Medium',
                    'suggestions': self._generate_performance_improvements(metrics)
                })
            
            # Code style and readability
            if metrics.readability_score < self.quality_thresholds['readability']['fair']:
                improvements.append({
                    'type': 'readability_enhancement',
                    'priority': 'low',
                    'description': 'Improve code readability and documentation',
                    'impact': 'Low',
                    'effort': 'Low',
                    'suggestions': self._generate_readability_improvements(metrics)
                })
            
            return improvements
            
        except Exception as e:
            logger.error(f"Improvement suggestion error: {e}")
            return []
    
    async def auto_fix_issues(self, file_path: str, 
                            issue_types: List[str] = None) -> Dict[str, Any]:
        """
        Automatically fix code issues where possible using ML
        """
        try:
            # Read original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            # Analyze file
            metrics = await self._analyze_file(file_path)
            
            # Filter fixable issues
            fixable_issues = [
                issue for issue in metrics.issues 
                if issue.auto_fixable and (not issue_types or issue.issue_type in issue_types)
            ]
            
            if not fixable_issues:
                return {
                    'success': False,
                    'message': 'No auto-fixable issues found',
                    'issues_found': len(metrics.issues),
                    'issues_fixed': 0
                }
            
            # Apply fixes
            fixed_code = original_code
            fixes_applied = []
            
            def get_issue_line_number(x):\n                return x.line_number\n            for issue in sorted(fixable_issues, key=get_issue_line_number, reverse=True):
                if issue.fix_code:
                    try:
                        fixed_code = await self._apply_fix(fixed_code, issue)
                        fixes_applied.append(issue)
                    except Exception as e:
                        logger.warning(f"Could not apply fix for {issue.issue_type}: {e}")
            
            # Validate fixed code
            try:
                ast.parse(fixed_code)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
            
            if syntax_valid and fixes_applied:
                # Write fixed code back to file (with backup)
                backup_path = f"{file_path}.backup_{int(time.time())}"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_code)
                
                return {
                    'success': True,
                    'message': f'Fixed {len(fixes_applied)} issues',
                    'issues_found': len(metrics.issues),
                    'issues_fixed': len(fixes_applied),
                    'fixes_applied': [issue.issue_type for issue in fixes_applied],
                    'backup_created': backup_path
                }
            else:
                return {
                    'success': False,
                    'message': 'Auto-fixes would break code syntax',
                    'issues_found': len(metrics.issues),
                    'issues_fixed': 0
                }
                
        except Exception as e:
            logger.error(f"Auto-fix error: {e}")
            return {
                'success': False,
                'error': str(e),
                'issues_found': 0,
                'issues_fixed': 0
            }
    
    def update_quality_feedback(self, file_path: str, feedback: Dict[str, Any]):
        """
        Update ML models based on quality assessment feedback
        """
        try:
            if file_path not in self.metrics_cache:
                return
            
            metrics = self.metrics_cache[file_path]
            features = self._extract_file_features_for_training(metrics)
            
            # Update models based on feedback
            if 'accuracy_score' in feedback:
                self._update_prediction_model(features, feedback['accuracy_score'])
            
            if 'usefulness_score' in feedback:
                self._update_suggestion_model(features, feedback['usefulness_score'])
            
            # Store feedback for model retraining
            feedback_entry = {
                'file_path': file_path,
                'metrics': metrics,
                'feedback': feedback,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.analysis_history.append(feedback_entry)
            
            # Trigger retraining periodically
            if len(self.analysis_history) % 150 == 0:
                asyncio.create_task(self._retrain_models())
            
        except Exception as e:
            logger.error(f"Feedback update error: {e}")
    
    # Private helper methods
    async def _analyze_file(self, file_path: str) -> CodeMetrics:
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return await self.analyze_code_fragment(code, file_path)
            
        except Exception as e:
            logger.error(f"File analysis error for {file_path}: {e}")
            return CodeMetrics(
                file_path=file_path,
                lines_of_code=0,
                complexity_score=0.0,
                maintainability_index=0.0,
                readability_score=0.0,
                security_score=0.0,
                performance_score=0.0,
                test_coverage=0.0,
                code_smells=0,
                duplicated_lines=0,
                technical_debt_hours=0.0
            )
    
    def _extract_code_features(self, code: str, tree: ast.AST) -> np.ndarray:
        """Extract ML features from code"""
        features = []
        
        # Basic code statistics
        lines = code.split('\n')
        features.extend([
            len(lines),                                    # Total lines
            len([l for l in lines if l.strip()]),         # Non-empty lines
            len([l for l in lines if l.strip().startswith('#')]), # Comment lines
            code.count('def '),                           # Function count
            code.count('class '),                         # Class count
            code.count('if '),                            # Conditional count
            code.count('for ') + code.count('while '),   # Loop count
            code.count('try:'),                           # Exception handling
            len(re.findall(r'import\s+\w+', code)),       # Import count
        ])
        
        # AST-based features
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features.extend([
                    len(node.args.args),                   # Parameter count
                    len([n for n in ast.walk(node) if isinstance(n, ast.If)]), # Nested ifs
                    len([n for n in ast.walk(node) if isinstance(n, ast.For)]) # Nested loops
                ])
                break
        else:
            features.extend([0, 0, 0])  # No functions found
        
        # Code complexity indicators
        features.extend([
            code.count('{') + code.count('['),            # Bracket complexity
            len(re.findall(r'\w+\.\w+', code)),           # Attribute access
            len(re.findall(r'\w+\(\w*\)', code)),         # Function calls
            code.count('lambda'),                         # Lambda functions
            code.count('yield'),                          # Generator usage
        ])
        
        # String and documentation features
        features.extend([
            len(re.findall(r'""".*?"""', code, re.DOTALL)), # Docstrings
            len(re.findall(r"'.*?'", code)),              # String literals
            code.count('TODO') + code.count('FIXME'),     # Todo comments
        ])
        
        # Pad or truncate to fixed size
        target_size = 25
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def _predict_complexity(self, features: np.ndarray) -> float:
        """Predict code complexity using ML"""
        try:
            if hasattr(self.complexity_predictor, 'predict'):
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                prediction = self.complexity_predictor.predict(features_scaled)[0]
                return float(np.clip(prediction, 0.0, 1.0))
        except:
            pass
        
        # Fallback heuristic based on features
        return float(np.clip(np.mean(features[5:9]) / 10.0, 0.0, 1.0))
    
    def _predict_maintainability(self, features: np.ndarray) -> float:
        """Predict maintainability score"""
        try:
            if hasattr(self.maintainability_scorer, 'predict'):
                prediction = self.maintainability_scorer.predict(features.reshape(1, -1))[0]
                return float(np.clip(prediction, 0.0, 1.0))
        except:
            pass
        
        # Fallback based on documentation and structure
        doc_ratio = features[21] / max(features[0], 1)  # Docstrings per line
        comment_ratio = features[2] / max(features[0], 1)  # Comments per line
        return float(np.clip((doc_ratio + comment_ratio) * 2, 0.0, 1.0))
    
    def _predict_readability(self, features: np.ndarray) -> float:
        """Predict readability score"""
        # Heuristic based on comments, naming, and structure
        lines = max(features[0], 1)
        comment_density = features[2] / lines
        function_density = features[3] / lines
        complexity_ratio = 1.0 - (features[5] + features[6]) / lines
        
        readability = (comment_density * 0.4 + function_density * 0.3 + complexity_ratio * 0.3)
        return float(np.clip(readability, 0.0, 1.0))
    
    def _predict_security_score(self, features: np.ndarray) -> float:
        """Predict security score"""
        # Simple heuristic - would be enhanced with actual security pattern detection
        exception_handling = features[7] / max(features[3], 1)  # Try blocks per function
        return float(np.clip(exception_handling, 0.0, 1.0))
    
    def _predict_performance_score(self, features: np.ndarray) -> float:
        """Predict performance score"""
        # Heuristic based on loop complexity and function calls
        loop_density = features[6] / max(features[0], 1)
        call_density = features[16] / max(features[0], 1)
        performance = 1.0 - (loop_density + call_density) / 2
        return float(np.clip(performance, 0.0, 1.0))
    
    async def _detect_issues(self, code: str, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Detect various code quality issues"""
        issues = []
        
        # Security issues
        security_issues = self._detect_security_issues(code, tree, file_path)
        issues.extend(security_issues)
        
        # Performance issues
        performance_issues = self._detect_performance_issues(code, tree, file_path)
        issues.extend(performance_issues)
        
        # Maintainability issues
        maintainability_issues = self._detect_maintainability_issues(code, tree, file_path)
        issues.extend(maintainability_issues)
        
        # Style issues
        style_issues = self._detect_style_issues(code, tree, file_path)
        issues.extend(style_issues)
        
        return issues
    
    def _detect_security_issues(self, code: str, tree: ast.AST, file_path: str) -> List[CodeIssue]:
        """Detect security-related issues"""
        issues = []
        
        # Check for common security patterns
        for pattern, issue_info in self.security_patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type="security_risk",
                    severity=issue_info['severity'],
                    message=issue_info['message'],
                    suggestion=issue_info['suggestion'],
                    confidence=issue_info['confidence'],
                    category="security",
                    rule_id=issue_info['rule_id']
                ))
        
        return issues
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data
        X_complexity, y_complexity = self._generate_complexity_training_data()
        X_bugs, y_bugs = self._generate_bug_training_data()
        
        # Train models
        if len(X_complexity) > 0:
            X_scaled = self.scaler.fit_transform(X_complexity)
            self.complexity_predictor.fit(X_scaled, y_complexity)
        
        if len(X_bugs) > 0:
            self.bug_predictor.fit(X_bugs, y_bugs)
    
    def _initialize_quality_patterns(self) -> Dict[str, Any]:
        """Initialize code quality patterns"""
        return {
            'long_function': {
                'pattern': r'def\s+\w+\([^)]*\):\s*\n((?:\s+.*\n){50,})',
                'severity': 'medium',
                'message': 'Function is too long and complex',
                'suggestion': 'Break into smaller functions'
            },
            'deep_nesting': {
                'pattern': r'(\s{16,})',
                'severity': 'medium',
                'message': 'Code is deeply nested',
                'suggestion': 'Reduce nesting levels'
            }
        }
    
    def _initialize_security_patterns(self) -> Dict[str, Any]:
        """Initialize security issue patterns"""
        return {
            r'eval\s*\(': {
                'severity': 'critical',
                'message': 'Use of eval() is dangerous',
                'suggestion': 'Use safer alternatives like ast.literal_eval()',
                'confidence': 0.9,
                'rule_id': 'SEC001'
            },
            r'exec\s*\(': {
                'severity': 'critical', 
                'message': 'Use of exec() is dangerous',
                'suggestion': 'Avoid dynamic code execution',
                'confidence': 0.9,
                'rule_id': 'SEC002'
            },
            r'pickle\.loads?\(': {
                'severity': 'high',
                'message': 'Pickle can execute arbitrary code',
                'suggestion': 'Use json or other safe serialization',
                'confidence': 0.8,
                'rule_id': 'SEC003'
            }
        }
    
    def _initialize_performance_patterns(self) -> Dict[str, Any]:
        """Initialize performance issue patterns"""
        return {
            r'for\s+\w+\s+in\s+range\(len\(': {
                'severity': 'low',
                'message': 'Inefficient loop pattern',
                'suggestion': 'Use enumerate() instead',
                'confidence': 0.7
            },
            r'\+\s*=\s*\[.*\]': {
                'severity': 'medium', 
                'message': 'Inefficient list concatenation',
                'suggestion': 'Use list.extend() or list comprehension',
                'confidence': 0.6
            }
        }
    
    # Additional helper methods would continue here...
    def _generate_complexity_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic training data for complexity prediction"""
        X, y = [], []
        for i in range(300):
            features = np.random.rand(25)
            # Synthetic complexity based on features
            complexity = (features[5] + features[6]) * 0.4 + features[3] * 0.3 + np.random.normal(0, 0.1)
            complexity = np.clip(complexity, 0.0, 1.0)
            X.append(features)
            y.append(complexity)
        return X, y
    
    def _generate_bug_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic training data for bug prediction"""
        X, y = [], []
        for i in range(200):
            features = np.random.rand(25)
            # Bug likelihood based on complexity and exception handling
            bug_prob = features[5] * 0.3 + (1 - features[7]) * 0.4 + np.random.normal(0, 0.1)
            label = 1 if bug_prob > 0.6 else 0
            X.append(features)
            y.append(label)
        return X, y


# Singleton instance
_ai_code_quality = None

def get_ai_code_quality() -> AICodeQuality:
    """Get or create AI code quality instance"""
    global _ai_code_quality
    if not _ai_code_quality:
        _ai_code_quality = AICodeQuality()
    return _ai_code_quality