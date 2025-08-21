import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from a2aNetwork.core.lnnQualityMonitor import LNNQualityMonitor, QualityMetric, QualityReport

# Mock Clients
class MockGrokClient:
    async def analyze(self, prompt: str):
        # Simulate a high-quality response
        return json.dumps({
            "accuracy_score": 95,
            "methodology_score": 90,
            "explanation_score": 88,
            "overall_score": 92.1,
            "confidence": 0.98
        })

class MockLNNClient:
    def __init__(self, is_trained=True, quality='good'):
        self.is_trained = is_trained
        self.quality = quality

    async def analyze(self, prompt: str):
        if not self.is_trained:
            return json.dumps({"overall_score": 20})
        
        if self.quality == 'good':
            # Simulate a slightly lower but acceptable quality response
            return json.dumps({
                "accuracy_score": 88,
                "methodology_score": 82,
                "explanation_score": 80,
                "overall_score": 84.4,
                "confidence": 0.91
            })
        else: # poor quality
            return json.dumps({
                "accuracy_score": 60,
                "methodology_score": 55,
                "explanation_score": 50,
                "overall_score": 56.5,
                "confidence": 0.7
            })

@pytest.fixture
def mock_grok_client():
    return MockGrokClient()

@pytest.fixture
def mock_lnn_client_good():
    return MockLNNClient(quality='good')

@pytest.fixture
def mock_lnn_client_poor():
    return MockLNNClient(quality='poor')

@pytest.fixture
def quality_monitor(mock_grok_client, mock_lnn_client_good):
    return LNNQualityMonitor(mock_grok_client, mock_lnn_client_good)

@pytest.mark.asyncio
async def test_run_single_benchmark(quality_monitor):
    """Test that a single benchmark runs and calculates deltas correctly."""
    test_case = quality_monitor.benchmark_tests[0]
    await quality_monitor._run_single_benchmark(test_case)
    
    assert len(quality_monitor.quality_history) == 1
    metric = quality_monitor.quality_history[0]
    
    assert metric.test_case_id == test_case['id']
    assert metric.accuracy_delta == 95 - 88
    assert metric.methodology_delta == 90 - 82
    assert metric.explanation_delta == 88 - 80
    assert pytest.approx(metric.overall_delta, 0.1) == 92.1 - 84.4

@pytest.mark.asyncio
async def test_generate_quality_report_good_quality(quality_monitor):
    """Test report generation with good quality LNN."""
    # Run a few benchmarks to populate history
    for test_case in quality_monitor.benchmark_tests:
        await quality_monitor._run_single_benchmark(test_case)
    
    report = await quality_monitor._generate_quality_report()
    
    assert report is not None
    assert report.quality_grade == 'B'
    assert report.acceptable_quality is True
    assert report.trend_direction == "insufficient_data" # Not enough data for trend

@pytest.mark.asyncio
async def test_generate_quality_report_poor_quality(mock_grok_client, mock_lnn_client_poor):
    """Test report generation with poor quality LNN."""
    monitor = LNNQualityMonitor(mock_grok_client, mock_lnn_client_poor)
    for test_case in monitor.benchmark_tests:
        await monitor._run_single_benchmark(test_case)
        
    report = await monitor._generate_quality_report()
    
    assert report.quality_grade == 'F'
    assert report.acceptable_quality is False
    assert "URGENT" in report.recommendations[0]

@pytest.mark.asyncio
async def test_monitoring_loop(quality_monitor):
    """Test that the monitoring loop starts, runs a cycle, and stops."""
    with patch.object(quality_monitor, '_run_benchmark_cycle', new_callable=AsyncMock) as mock_run_cycle:
        quality_monitor.config['monitoring_interval'] = 0.1
        await quality_monitor.start_monitoring()
        await asyncio.sleep(0.2) # Allow loop to run at least once
        await quality_monitor.stop_monitoring()
        
        assert mock_run_cycle.called
        assert not quality_monitor.is_monitoring

@pytest.mark.asyncio
async def test_quality_alert(quality_monitor):
    """Test that a quality alert is triggered for large deltas."""
    quality_monitor.config['alert_threshold'] = 10.0
    lnn_client_poor = MockLNNClient(quality='poor')
    quality_monitor.lnn_client = lnn_client_poor

    with patch('a2aNetwork.core.lnnQualityMonitor.logger.warning') as mock_log_warning:
        test_case = quality_monitor.benchmark_tests[0]
        await quality_monitor._run_single_benchmark(test_case)
        
        mock_log_warning.assert_called_once()
        call_args = mock_log_warning.call_args[0][0]
        assert "LNN QUALITY ALERT" in call_args
        assert f"Delta: {92.1 - 56.5:.1f}" in call_args
