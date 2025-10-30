# services/config_manager.py
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from loguru import logger
from threading import Lock
import yaml

from app.config import settings
from app.models.schemas import ExtractionResult


class DocumentType(Enum):
    """Types of documents for specialized processing"""
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_REPORT = "technical_report"
    BOOK_CHAPTER = "book_chapter"
    THESIS = "thesis"
    PREPRINT = "preprint"
    CONFERENCE_PAPER = "conference_paper"
    JOURNAL_ARTICLE = "journal_article"
    UNKNOWN = "unknown"


class ProcessingProfile(Enum):
    """Processing profiles for different use cases"""
    FAST = "fast"           # Speed optimized
    BALANCED = "balanced"   # Balance of speed and quality
    QUALITY = "quality"     # Quality optimized
    COMPREHENSIVE = "comprehensive"  # Maximum extraction
    MINIMAL = "minimal"     # Minimal processing


@dataclass
class ExtractionConfig:
    """Dynamic configuration for extraction methods"""
    
    # Method selection and weights
    grobid_enabled: bool = True
    grobid_weight: float = 0.9
    grobid_timeout: int = 120
    
    table_transformer_enabled: bool = True
    table_transformer_weight: float = 0.8
    table_validation_threshold: float = 0.75
    

    figure_validation_threshold: float = 0.7
    
    ocr_enabled: bool = True
    ocr_weight: float = 0.6
    ocr_fallback_threshold: float = 0.5
    
    cv_detection_enabled: bool = False
    cv_detection_weight: float = 0.6
    
    # Quality thresholds
    min_acceptable_quality: float = 0.6
    target_quality: float = 0.8
    retry_quality_threshold: float = 0.7
    
    # Performance constraints
    max_processing_time: int = 600  # seconds
    memory_limit_mb: int = 4096
    parallel_extraction: bool = True
    
    # Content-specific settings
    table_false_positive_threshold: float = 0.3
    text_coherence_threshold: float = 0.6
    figure_detection_threshold: float = 0.8
    
    # Retry and fallback settings
    max_retries: int = 2
    retry_delay: int = 30
    fallback_to_ocr: bool = True
    enable_error_recovery: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class PerformanceMetrics:
    """Performance metrics for configuration optimization"""
    success_rate: float = 0.0
    average_quality: float = 0.0
    average_processing_time: float = 0.0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    
    # Method-specific metrics
    method_success_rates: Dict[str, float] = field(default_factory=dict)
    method_avg_times: Dict[str, float] = field(default_factory=dict)
    
    # Quality breakdown
    table_accuracy: float = 0.0
    figure_accuracy: float = 0.0
    text_coherence: float = 0.0
    extraction_coverage: float = 0.0
    
    sample_size: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConfigurationRule:
    """Rule for dynamic configuration adjustment"""
    name: str
    condition: str  # Python expression
    action: str     # Configuration change action
    priority: int = 0
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition"""
        try:
            return eval(self.condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"Rule evaluation failed for {self.name}: {e}")
            return False


class ConfigurationManager:
    """
    Dynamic configuration management for extraction pipeline
    Automatically adjusts parameters based on performance feedback
    """
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or settings.paper_folder / "config"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Current configurations
        self.configs: Dict[str, ExtractionConfig] = {}
        self.default_config = ExtractionConfig()
        
        # Performance tracking
        self.performance_history: Dict[str, List[PerformanceMetrics]] = {}
        self.current_metrics: Dict[str, PerformanceMetrics] = {}
        
        # Configuration rules
        self.rules: List[ConfigurationRule] = []
        
        # Thread safety
        self.lock = Lock()
        
        # Auto-optimization settings
        self.auto_optimization_enabled = True
        self.optimization_interval = timedelta(hours=1)
        self.last_optimization = datetime.utcnow()
        
        # Initialize (will be called when needed)
        self._initialized = False
    
    async def _initialize_configs(self):
        """Initialize configuration system"""
        try:
            # Load existing configurations
            await self._load_configurations()
            
            # Load performance history
            await self._load_performance_history()
            
            # Load configuration rules
            await self._load_configuration_rules()
            
            # Start optimization task
            if self.auto_optimization_enabled:
                asyncio.create_task(self._auto_optimization_loop())
                
            logger.info("Configuration manager initialized")
            
        except Exception as e:
            logger.error(f"Configuration manager initialization failed: {e}")
    
    async def _load_configurations(self):
        """Load configurations from storage"""
        config_files = {
            'default': self.config_dir / "default_config.json",
            'academic_paper': self.config_dir / "academic_paper_config.json",
            'technical_report': self.config_dir / "technical_report_config.json",
            'fast': self.config_dir / "fast_profile_config.json",
            'quality': self.config_dir / "quality_profile_config.json",
        }
        
        for profile_name, config_path in config_files.items():
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    self.configs[profile_name] = ExtractionConfig.from_dict(config_data)
                    logger.info(f"Loaded configuration for {profile_name}")
                except Exception as e:
                    logger.warning(f"Failed to load config {profile_name}: {e}")
            else:
                # Create default configuration
                self.configs[profile_name] = self._create_default_config(profile_name)
                await self._save_configuration(profile_name)
    
    def _create_default_config(self, profile_name: str) -> ExtractionConfig:
        """Create default configuration for profile"""
        base_config = ExtractionConfig()
        
        if profile_name == 'fast':
            # Fast profile: prioritize speed
            base_config.ocr_enabled = False
            base_config.cv_detection_enabled = False
            base_config.max_processing_time = 300
            base_config.max_retries = 1
            base_config.parallel_extraction = True
            base_config.target_quality = 0.6
            
        elif profile_name == 'quality':
            # Quality profile: prioritize accuracy
            base_config.ocr_enabled = True
            base_config.cv_detection_enabled = True
            base_config.max_processing_time = 900
            base_config.max_retries = 3
            base_config.target_quality = 0.9
            base_config.table_validation_threshold = 0.8
            base_config.figure_validation_threshold = 0.8
            
        elif profile_name == 'academic_paper':
            # Academic paper optimized
            
            base_config.table_transformer_enabled = True
            base_config.target_quality = 0.8
            
        elif profile_name == 'technical_report':
            # Technical report optimized
            base_config.cv_detection_enabled = True
            base_config.table_validation_threshold = 0.7
            base_config.target_quality = 0.75
        
        return base_config
    
    async def _save_configuration(self, profile_name: str):
        """Save configuration to storage"""
        if profile_name in self.configs:
            config_path = self.config_dir / f"{profile_name}_config.json"
            try:
                with open(config_path, 'w') as f:
                    json.dump(self.configs[profile_name].to_dict(), f, indent=2, default=str)
                logger.debug(f"Saved configuration for {profile_name}")
            except Exception as e:
                logger.error(f"Failed to save config {profile_name}: {e}")
    
    async def _load_performance_history(self):
        """Load performance history from storage"""
        history_path = self.config_dir / "performance_history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                
                for profile, metrics_list in data.items():
                    self.performance_history[profile] = []
                    for metrics_data in metrics_list:
                        metrics = PerformanceMetrics(**metrics_data)
                        metrics.last_updated = datetime.fromisoformat(metrics_data['last_updated'])
                        self.performance_history[profile].append(metrics)
                
                logger.info("Loaded performance history")
            except Exception as e:
                logger.warning(f"Failed to load performance history: {e}")
    
    async def _save_performance_history(self):
        """Save performance history to storage"""
        history_path = self.config_dir / "performance_history.json"
        try:
            data = {}
            for profile, metrics_list in self.performance_history.items():
                data[profile] = []
                for metrics in metrics_list[-50:]:  # Keep last 50 records
                    metrics_dict = asdict(metrics)
                    metrics_dict['last_updated'] = metrics.last_updated.isoformat()
                    data[profile].append(metrics_dict)
            
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")
    
    async def _load_configuration_rules(self):
        """Load configuration rules"""
        rules_path = self.config_dir / "configuration_rules.yaml"
        
        if not rules_path.exists():
            # Create default rules
            await self._create_default_rules()
        
        try:
            with open(rules_path, 'r') as f:
                rules_data = yaml.safe_load(f)
            
            self.rules = []
            for rule_data in rules_data.get('rules', []):
                rule = ConfigurationRule(**rule_data)
                self.rules.append(rule)
            
            logger.info(f"Loaded {len(self.rules)} configuration rules")
            
        except Exception as e:
            logger.error(f"Failed to load configuration rules: {e}")
    
    async def _create_default_rules(self):
        """Create default configuration rules"""
        default_rules = [
            ConfigurationRule(
                name="high_table_fp_rate",
                condition="metrics.get('table_false_positive_rate', 0) > 0.4",
                action="increase_table_validation_threshold",
                priority=10,
                description="Increase table validation threshold when false positive rate is high"
            ),
            ConfigurationRule(
                name="low_text_coherence",
                condition="metrics.get('text_coherence', 1) < 0.5",
                action="enable_ocr_fallback",
                priority=8,
                description="Enable OCR fallback when text coherence is low"
            ),
            ConfigurationRule(
                name="slow_processing",
                condition="metrics.get('average_processing_time', 0) > 300",
                action="disable_cv_detection",
                priority=6,
                description="Disable CV detection when processing is slow"
            ),
            ConfigurationRule(
                name="low_success_rate",
                condition="metrics.get('success_rate', 1) < 0.8",
                action="enable_error_recovery",
                priority=9,
                description="Enable enhanced error recovery when success rate is low"
            ),
            ConfigurationRule(
                name="high_quality_performance",
                condition="metrics.get('average_quality', 0) > 0.85 and metrics.get('success_rate', 0) > 0.9",
                action="optimize_for_speed",
                priority=4,
                description="Optimize for speed when quality and success rate are high"
            ),
            ConfigurationRule(
                name="memory_pressure",
                condition="metrics.get('memory_usage', 0) > 3000",  # MB
                action="reduce_parallel_processing",
                priority=7,
                description="Reduce parallel processing under memory pressure"
            )
        ]
        
        rules_data = {
            'rules': [asdict(rule) for rule in default_rules]
        }
        
        rules_path = self.config_dir / "configuration_rules.yaml"
        with open(rules_path, 'w') as f:
            yaml.dump(rules_data, f, default_flow_style=False)
    
    async def _ensure_initialized(self):
        """Ensure configuration system is initialized"""
        if not self._initialized:
            await self._initialize_configs()
            self._initialized = True

    async def get_optimal_config(self, 
                               document_type: DocumentType = DocumentType.UNKNOWN,
                               processing_profile: ProcessingProfile = ProcessingProfile.BALANCED,
                               quality_target: float = None,
                               time_constraint: int = None,
                               context: Dict[str, Any] = None) -> ExtractionConfig:
        """
        Get optimal configuration based on document type, profile, and context
        """
        await self._ensure_initialized()
        
        with self.lock:
            # Start with base configuration
            base_config_name = self._select_base_config(document_type, processing_profile)
            config = self.configs.get(base_config_name, self.default_config)
            
            # Clone the configuration
            config_dict = config.to_dict()
            adapted_config = ExtractionConfig.from_dict(config_dict)
            
            # Apply user constraints
            if quality_target is not None:
                adapted_config.target_quality = quality_target
                adapted_config.min_acceptable_quality = max(0.4, quality_target - 0.2)
            
            if time_constraint is not None:
                adapted_config.max_processing_time = time_constraint
                
                # Adapt other settings based on time constraint
                if time_constraint < 180:  # 3 minutes
                    adapted_config.ocr_enabled = False
                    adapted_config.cv_detection_enabled = False
                    adapted_config.max_retries = 1
                elif time_constraint < 300:  # 5 minutes
                    adapted_config.cv_detection_enabled = False
                    adapted_config.max_retries = 2
            
            # Apply performance-based adaptations
            adapted_config = self._apply_performance_adaptations(adapted_config, base_config_name)
            
            # Apply dynamic rules with proper context
            rule_context = context or {}
            # Add current metrics to context for rule evaluation
            if base_config_name in self.current_metrics:
                rule_context['metrics'] = {
                    'table_false_positive_rate': self.current_metrics[base_config_name].table_accuracy,
                    'text_coherence': self.current_metrics[base_config_name].text_coherence,
                    'average_processing_time': self.current_metrics[base_config_name].average_processing_time,
                    'success_rate': self.current_metrics[base_config_name].success_rate,
                    'average_quality': self.current_metrics[base_config_name].average_quality,
                    'memory_usage': 0.0  # Default value
                }
            else:
                # Provide default metrics for rule evaluation
                rule_context['metrics'] = {
                    'table_false_positive_rate': 0.0,
                    'text_coherence': 1.0,
                    'average_processing_time': 0.0,
                    'success_rate': 1.0,
                    'average_quality': 1.0,
                    'memory_usage': 0.0
                }
            
            adapted_config = self._apply_configuration_rules(adapted_config, rule_context)
            
            return adapted_config
    
    def _select_base_config(self, document_type: DocumentType, 
                          processing_profile: ProcessingProfile) -> str:
        """Select base configuration name"""
        
        # Profile takes precedence
        if processing_profile == ProcessingProfile.FAST:
            return 'fast'
        elif processing_profile == ProcessingProfile.QUALITY:
            return 'quality'
        elif processing_profile == ProcessingProfile.COMPREHENSIVE:
            return 'quality'  # Use quality config for comprehensive
        elif processing_profile == ProcessingProfile.MINIMAL:
            return 'fast'     # Use fast config for minimal
        
        # Document type specific
        if document_type == DocumentType.ACADEMIC_PAPER:
            return 'academic_paper'
        elif document_type == DocumentType.TECHNICAL_REPORT:
            return 'technical_report'
        
        return 'default'
    
    def _apply_performance_adaptations(self, config: ExtractionConfig, 
                                     base_config_name: str) -> ExtractionConfig:
        """Apply adaptations based on performance history"""
        
        if base_config_name not in self.performance_history:
            return config
        
        recent_metrics = self.performance_history[base_config_name][-10:]  # Last 10 records
        if not recent_metrics:
            return config
        
        # Calculate average metrics
        avg_quality = np.mean([m.average_quality for m in recent_metrics])
        avg_success_rate = np.mean([m.success_rate for m in recent_metrics])
        avg_processing_time = np.mean([m.average_processing_time for m in recent_metrics])
        
        # Adapt based on performance patterns
        if avg_quality < 0.6:
            # Low quality - enable more methods
            config.ocr_enabled = True
            config.cv_detection_enabled = True
            config.max_retries = min(config.max_retries + 1, 3)
            
        elif avg_quality > 0.85 and avg_success_rate > 0.9:
            # High quality and success - optimize for speed
            if avg_processing_time > 300:
                config.cv_detection_enabled = False
                config.max_retries = max(config.max_retries - 1, 1)
        
        if avg_success_rate < 0.8:
            # Low success rate - be more conservative
            config.enable_error_recovery = True
            config.fallback_to_ocr = True
            config.retry_delay = max(config.retry_delay, 60)
        
        if avg_processing_time > 450:  # 7.5 minutes
            # Slow processing - disable expensive methods
            config.cv_detection_enabled = False
            config.ocr_enabled = False
            config.parallel_extraction = False
        
        return config
    
    def _apply_configuration_rules(self, config: ExtractionConfig, 
                                 context: Dict[str, Any]) -> ExtractionConfig:
        """Apply dynamic configuration rules"""
        
        # Sort rules by priority (higher priority first)
        sorted_rules = sorted([r for r in self.rules if r.enabled], 
                             key=lambda x: x.priority, reverse=True)
        
        for rule in sorted_rules:
            try:
                if rule.evaluate(context):
                    config = self._apply_rule_action(config, rule.action, context)
                    logger.debug(f"Applied rule: {rule.name}")
            except Exception as e:
                logger.warning(f"Failed to apply rule {rule.name}: {e}")
        
        return config
    
    def _apply_rule_action(self, config: ExtractionConfig, action: str, 
                          context: Dict[str, Any]) -> ExtractionConfig:
        """Apply configuration rule action"""
        
        if action == "increase_table_validation_threshold":
            config.table_validation_threshold = min(0.9, config.table_validation_threshold + 0.1)
            
        elif action == "enable_ocr_fallback":
            config.ocr_enabled = True
            config.fallback_to_ocr = True
            
        elif action == "disable_cv_detection":
            config.cv_detection_enabled = False
            
        elif action == "enable_error_recovery":
            config.enable_error_recovery = True
            config.max_retries = max(config.max_retries, 2)
            
        elif action == "optimize_for_speed":
            config.cv_detection_enabled = False
            config.max_retries = 1
            config.ocr_enabled = False
            
        elif action == "reduce_parallel_processing":
            config.parallel_extraction = False
            config.memory_limit_mb = min(config.memory_limit_mb, 2048)
        
        return config
    
    async def update_performance_metrics(self, config_name: str, 
                                       extraction_result: ExtractionResult,
                                       processing_time: float,
                                       success: bool,
                                       error_count: int = 0):
        """Update performance metrics for configuration optimization"""
        
        with self.lock:
            if config_name not in self.performance_history:
                self.performance_history[config_name] = []
            
            # Extract quality metrics
            quality_metrics = getattr(extraction_result, 'quality_metrics', None)
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                success_rate=1.0 if success else 0.0,
                average_quality=quality_metrics.overall_score if quality_metrics else 0.0,
                average_processing_time=processing_time,
                error_rate=error_count,
                table_accuracy=quality_metrics.table_accuracy if quality_metrics else 0.0,
                figure_accuracy=quality_metrics.figure_accuracy if quality_metrics else 0.0,
                text_coherence=quality_metrics.text_coherence if quality_metrics else 0.0,
                extraction_coverage=quality_metrics.extraction_coverage if quality_metrics else 0.0,
                sample_size=1
            )
            
            # Extract method-specific metrics
            if extraction_result.extraction_methods:
                for method in extraction_result.extraction_methods:
                    metrics.method_success_rates[method] = 1.0 if success else 0.0
                    metrics.method_avg_times[method] = processing_time / len(extraction_result.extraction_methods)
            
            # Add to history
            self.performance_history[config_name].append(metrics)
            
            # Keep only recent history (last 100 records)
            if len(self.performance_history[config_name]) > 100:
                self.performance_history[config_name] = self.performance_history[config_name][-100:]
            
            # Update current metrics (rolling average)
            self._update_current_metrics(config_name)
            
            # Save to storage periodically
            if len(self.performance_history[config_name]) % 10 == 0:
                await self._save_performance_history()
    
    def _update_current_metrics(self, config_name: str):
        """Update current metrics with rolling average"""
        
        if config_name not in self.performance_history:
            return
        
        recent_metrics = self.performance_history[config_name][-20:]  # Last 20 records
        
        if not recent_metrics:
            return
        
        # Calculate rolling averages
        current = PerformanceMetrics(
            success_rate=np.mean([m.success_rate for m in recent_metrics]),
            average_quality=np.mean([m.average_quality for m in recent_metrics]),
            average_processing_time=np.mean([m.average_processing_time for m in recent_metrics]),
            error_rate=np.mean([m.error_rate for m in recent_metrics]),
            table_accuracy=np.mean([m.table_accuracy for m in recent_metrics]),
            figure_accuracy=np.mean([m.figure_accuracy for m in recent_metrics]),
            text_coherence=np.mean([m.text_coherence for m in recent_metrics]),
            extraction_coverage=np.mean([m.extraction_coverage for m in recent_metrics]),
            sample_size=len(recent_metrics)
        )
        
        self.current_metrics[config_name] = current
    
    async def _auto_optimization_loop(self):
        """Background task for automatic configuration optimization"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if datetime.utcnow() - self.last_optimization > self.optimization_interval:
                    await self._optimize_configurations()
                    self.last_optimization = datetime.utcnow()
                    
            except Exception as e:
                logger.error(f"Auto-optimization failed: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _optimize_configurations(self):
        """Optimize configurations based on performance data"""
        logger.info("Starting automatic configuration optimization")
        
        with self.lock:
            for config_name, metrics_history in self.performance_history.items():
                if len(metrics_history) < 10:  # Need sufficient data
                    continue
                
                try:
                    await self._optimize_single_config(config_name, metrics_history)
                except Exception as e:
                    logger.error(f"Failed to optimize config {config_name}: {e}")
        
        # Save optimized configurations
        for config_name in self.configs:
            await self._save_configuration(config_name)
        
        logger.info("Configuration optimization completed")
    
    async def _optimize_single_config(self, config_name: str, 
                                    metrics_history: List[PerformanceMetrics]):
        """Optimize a single configuration based on its performance history"""
        
        if config_name not in self.configs:
            return
        
        config = self.configs[config_name]
        recent_metrics = metrics_history[-20:]
        
        # Calculate performance indicators
        avg_quality = np.mean([m.average_quality for m in recent_metrics])
        avg_success_rate = np.mean([m.success_rate for m in recent_metrics])
        avg_processing_time = np.mean([m.average_processing_time for m in recent_metrics])
        avg_table_accuracy = np.mean([m.table_accuracy for m in recent_metrics])
        avg_figure_accuracy = np.mean([m.figure_accuracy for m in recent_metrics])
        
        # Optimization strategies
        optimization_made = False
        
        # 1. Table accuracy optimization
        if avg_table_accuracy < 0.7 and config.table_validation_threshold < 0.8:
            config.table_validation_threshold = min(0.85, config.table_validation_threshold + 0.05)
            optimization_made = True
            logger.info(f"Increased table validation threshold for {config_name}")
        
        # 2. Processing time optimization
        if avg_processing_time > 400 and avg_quality > 0.8:
            # Good quality but slow - optimize for speed
            if config.cv_detection_enabled:
                config.cv_detection_enabled = False
                optimization_made = True
                logger.info(f"Disabled CV detection for {config_name} (speed optimization)")
        
        # 3. Quality optimization
        if avg_quality < 0.7 and avg_success_rate > 0.8:
            # Low quality but successful - enable more methods
            if not config.ocr_enabled:
                config.ocr_enabled = True
                optimization_made = True
                logger.info(f"Enabled OCR for {config_name} (quality optimization)")
        
        # 4. Success rate optimization
        if avg_success_rate < 0.8:
            # Low success rate - be more conservative
            config.max_retries = min(3, config.max_retries + 1)
            config.enable_error_recovery = True
            optimization_made = True
            logger.info(f"Increased retries for {config_name} (success rate optimization)")
        
        # 5. Figure detection optimization
        if avg_figure_accuracy < 0.6 and not config.cv_detection_enabled:
            config.cv_detection_enabled = True
            optimization_made = True
            logger.info(f"Enabled CV detection for {config_name} (figure accuracy optimization)")
        
        if optimization_made:
            logger.info(f"Optimized configuration for {config_name}")
    
    async def get_configuration_report(self) -> Dict[str, Any]:
        """Get comprehensive configuration and performance report"""
        
        await self._ensure_initialized()
        
        with self.lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "configurations": {},
                "performance_summary": {},
                "optimization_recommendations": [],
                "rules_status": []
            }
            
            # Configuration summary
            for config_name, config in self.configs.items():
                report["configurations"][config_name] = {
                    "config": config.to_dict(),
                    "last_updated": getattr(config, 'last_updated', 'unknown')
                }
            
            # Performance summary
            for config_name, current_metrics in self.current_metrics.items():
                report["performance_summary"][config_name] = asdict(current_metrics)
            
            # Optimization recommendations
            recommendations = await self._generate_optimization_recommendations()
            report["optimization_recommendations"] = recommendations
            
            # Rules status
            for rule in self.rules:
                report["rules_status"].append({
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "description": rule.description,
                    "priority": rule.priority
                })
            
            return report
    
    async def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on current performance"""
        
        recommendations = []
        
        for config_name, metrics in self.current_metrics.items():
            if metrics.sample_size < 5:
                continue
            
            config = self.configs.get(config_name)
            if not config:
                continue
            
            # Quality recommendations
            if metrics.average_quality < 0.7:
                recommendations.append(
                    f"{config_name}: Consider enabling additional extraction methods to improve quality "
                    f"(current: {metrics.average_quality:.2f})"
                )
            
            # Performance recommendations
            if metrics.average_processing_time > 400:
                recommendations.append(
                    f"{config_name}: Processing time is high ({metrics.average_processing_time:.1f}s). "
                    f"Consider disabling non-essential methods"
                )
            
            # Success rate recommendations
            if metrics.success_rate < 0.8:
                recommendations.append(
                    f"{config_name}: Success rate is low ({metrics.success_rate:.1%}). "
                    f"Consider enabling error recovery and increasing retries"
                )
            
            # Method-specific recommendations
            if metrics.table_accuracy < 0.7:
                recommendations.append(
                    f"{config_name}: Table accuracy is low ({metrics.table_accuracy:.2f}). "
                    f"Consider adjusting table validation threshold"
                )
            
            if metrics.figure_accuracy < 0.6:
                recommendations.append(
                    f"{config_name}: Figure accuracy is low ({metrics.figure_accuracy:.2f}). "
                    f"Consider enabling computer vision detection"
                )
        
        return recommendations
    
    async def add_configuration_rule(self, rule: ConfigurationRule):
        """Add new configuration rule"""
        
        with self.lock:
            # Check if rule with same name exists
            existing_rule_idx = None
            for i, existing_rule in enumerate(self.rules):
                if existing_rule.name == rule.name:
                    existing_rule_idx = i
                    break
            
            if existing_rule_idx is not None:
                # Update existing rule
                self.rules[existing_rule_idx] = rule
                logger.info(f"Updated configuration rule: {rule.name}")
            else:
                # Add new rule
                self.rules.append(rule)
                logger.info(f"Added configuration rule: {rule.name}")
            
            # Save rules
            await self._save_configuration_rules()
    
    async def _save_configuration_rules(self):
        """Save configuration rules to storage"""
        rules_path = self.config_dir / "configuration_rules.yaml"
        
        rules_data = {
            'rules': [asdict(rule) for rule in self.rules]
        }
        
        try:
            with open(rules_path, 'w') as f:
                yaml.dump(rules_data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save configuration rules: {e}")
    
    def get_config_for_extraction_request(self, message: Dict[str, Any]) -> ExtractionConfig:
        """Get configuration for specific extraction request"""
        
        # Extract parameters from message
        quality_target = message.get('qualityTarget', 0.75)
        time_constraint = message.get('timeConstraint')
        document_hints = message.get('documentHints', {})
        processing_profile = ProcessingProfile(message.get('processingProfile', 'balanced'))
        
        # Determine document type from hints
        document_type = DocumentType.UNKNOWN
        if document_hints:
            doc_type_str = document_hints.get('type', 'unknown')
            try:
                document_type = DocumentType(doc_type_str)
            except ValueError:
                document_type = DocumentType.UNKNOWN
        
        # Build context
        context = {
            'message': message,
            'document_hints': document_hints,
            'metrics': self.current_metrics.get('default', {})
        }
        
        # Get optimal configuration
        return asyncio.run(self.get_optimal_config(
            document_type=document_type,
            processing_profile=processing_profile,
            quality_target=quality_target,
            time_constraint=time_constraint,
            context=context
        ))


# Global configuration manager instance
config_manager = ConfigurationManager()
