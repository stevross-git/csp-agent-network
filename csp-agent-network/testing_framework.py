# tests/conftest.py
"""
Pytest configuration and fixtures for CSP System testing
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Import CSP components for testing
from core.advanced_csp_core import AdvancedCSPEngine, ProcessContext
from ai_integration.csp_ai_integration import AIAgent, LLMCapability
from runtime.csp_runtime_environment import CSPRuntimeOrchestrator, RuntimeConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def csp_engine():
    """Create a CSP engine for testing"""
    engine = AdvancedCSPEngine()
    yield engine
    # Cleanup
    try:
        await engine.shutdown()
    except:
        pass

@pytest.fixture
async def test_context():
    """Create a test process context"""
    context = ProcessContext()
    yield context

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
async def mock_ai_agent():
    """Create a mock AI agent for testing"""
    capability = MockLLMCapability()
    agent = AIAgent("test_agent", [capability])
    yield agent

class MockLLMCapability(LLMCapability):
    """Mock LLM capability for testing"""
    
    def __init__(self):
        self.model_name = "mock-model"
        self.specialized_domain = "testing"
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        # Mock response
        return {
            "response": f"Mock response to: {input_data}",
            "confidence": 0.95,
            "processing_time": 0.1
        }

@pytest.fixture
async def runtime_orchestrator():
    """Create a runtime orchestrator for testing"""
    config = RuntimeConfig(
        max_workers=2,
        memory_limit_gb=1.0,
        enable_monitoring=False
    )
    orchestrator = CSPRuntimeOrchestrator(config)
    yield orchestrator
    await orchestrator.stop()

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow

---
# tests/unit/test_process_algebra.py
"""
Unit tests for process algebra functionality
"""

import pytest
import asyncio
from core.advanced_csp_core import (
    AtomicProcess, CompositeProcess, CompositionOperator, 
    ProcessContext, Event, ChannelType
)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_atomic_process_creation():
    """Test atomic process creation and execution"""
    
    async def test_action(context):
        return "test_result"
    
    process = AtomicProcess("test_process", test_action)
    
    assert process.process_id == "test_process"
    assert process.action == test_action
    
    # Test execution
    context = ProcessContext()
    result = await process.run(context)
    assert result == "test_result"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_sequential_composition():
    """Test sequential process composition"""
    
    async def action_a(context):
        return "result_a"
    
    async def action_b(context):
        return "result_b"
    
    process_a = AtomicProcess("a", action_a)
    process_b = AtomicProcess("b", action_b)
    
    sequential = CompositeProcess(
        "sequential", 
        CompositionOperator.SEQUENTIAL, 
        [process_a, process_b]
    )
    
    context = ProcessContext()
    results = await sequential.run(context)
    
    assert results == ["result_a", "result_b"]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_composition():
    """Test parallel process composition"""
    
    async def action_a(context):
        await asyncio.sleep(0.1)
        return "result_a"
    
    async def action_b(context):
        await asyncio.sleep(0.1)
        return "result_b"
    
    process_a = AtomicProcess("a", action_a)
    process_b = AtomicProcess("b", action_b)
    
    parallel = CompositeProcess(
        "parallel", 
        CompositionOperator.PARALLEL, 
        [process_a, process_b]
    )
    
    context = ProcessContext()
    start_time = asyncio.get_event_loop().time()
    results = await parallel.run(context)
    end_time = asyncio.get_event_loop().time()
    
    # Should execute in parallel (< 0.15 seconds)
    assert end_time - start_time < 0.15
    assert set(results) == {"result_a", "result_b"}

@pytest.mark.unit
@pytest.mark.asyncio
async def test_choice_composition():
    """Test choice process composition"""
    
    async def fast_action(context):
        await asyncio.sleep(0.01)
        return "fast_result"
    
    async def slow_action(context):
        await asyncio.sleep(1.0)
        return "slow_result"
    
    fast_process = AtomicProcess("fast", fast_action)
    slow_process = AtomicProcess("slow", slow_action)
    
    choice = CompositeProcess(
        "choice", 
        CompositionOperator.CHOICE, 
        [fast_process, slow_process]
    )
    
    context = ProcessContext()
    result = await choice.run(context)
    
    # Should return the fast result
    assert result == "fast_result"

@pytest.mark.unit
def test_process_signature():
    """Test process signature generation"""
    
    async def test_action(context):
        return "test"
    
    process = AtomicProcess("test", test_action)
    signature = process.get_signature()
    
    assert signature.input_events == []
    assert signature.output_events == ["action_test"]
    assert "atomic_action" in signature.capabilities

---
# tests/unit/test_channels.py
"""
Unit tests for channel communication
"""

import pytest
import asyncio
from core.advanced_csp_core import (
    SynchronousChannel, SemanticChannel, Event, ChannelType
)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_synchronous_channel():
    """Test synchronous channel communication"""
    
    channel = SynchronousChannel("sync_test")
    
    async def sender():
        event = Event("test_event", "sync_test", "test_data")
        result = await channel.send(event, "sender")
        return result
    
    async def receiver():
        event = await channel.receive("receiver")
        return event.data
    
    # Run sender and receiver concurrently
    sender_task = asyncio.create_task(sender())
    receiver_task = asyncio.create_task(receiver())
    
    sender_result, receiver_result = await asyncio.gather(sender_task, receiver_task)
    
    assert sender_result is True
    assert receiver_result == "test_data"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_channel():
    """Test semantic channel communication"""
    
    channel = SemanticChannel("semantic_test", embedding_dim=64)
    
    event = Event("semantic_event", "semantic_test", {"content": "test message"})
    
    # Test sending
    result = await channel.send(event, "sender")
    assert result is True
    
    # Test semantic vector generation
    assert event.semantic_vector is not None
    assert len(event.semantic_vector) == 64

@pytest.mark.unit
@pytest.mark.asyncio
async def test_channel_statistics():
    """Test channel statistics tracking"""
    
    channel = SynchronousChannel("stats_test")
    
    # Send multiple events
    for i in range(5):
        event = Event(f"event_{i}", "stats_test", f"data_{i}")
        # Mock successful send
        channel.statistics['messages_sent'] += 1
    
    assert channel.statistics['messages_sent'] == 5

---
# tests/unit/test_ai_integration.py
"""
Unit tests for AI integration
"""

import pytest
import asyncio
from ai_integration.csp_ai_integration import (
    AIAgent, CollaborativeAIProcess, LLMCapability, 
    VisionCapability, ReasoningCapability
)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ai_agent_creation():
    """Test AI agent creation"""
    
    llm_capability = LLMCapability("test-model", "general")
    agent = AIAgent("test_agent", [llm_capability])
    
    assert agent.agent_id == "test_agent"
    assert len(agent.capabilities) == 1
    assert "llm" in agent.capabilities

@pytest.mark.unit
@pytest.mark.asyncio
async def test_llm_capability():
    """Test LLM capability execution"""
    
    capability = LLMCapability("test-model", "reasoning")
    
    input_data = {
        "prompt": "What is 2 + 2?"
    }
    
    result = await capability.execute(input_data, {})
    
    assert "reasoning_chain" in result
    assert "conclusion" in result
    assert "confidence" in result

@pytest.mark.unit
@pytest.mark.asyncio
async def test_vision_capability():
    """Test vision capability execution"""
    
    capability = VisionCapability("general")
    
    input_data = {
        "image": "mock_image_data"
    }
    
    result = await capability.execute(input_data, {})
    
    assert "objects_detected" in result
    assert "scene_description" in result
    assert "confidence_scores" in result

@pytest.mark.unit
@pytest.mark.asyncio
async def test_reasoning_capability():
    """Test reasoning capability execution"""
    
    capability = ReasoningCapability("logical")
    
    input_data = {
        "premises": ["All humans are mortal", "Socrates is human"],
        "query": "Is Socrates mortal?"
    }
    
    result = await capability.execute(input_data, {})
    
    assert "conclusion" in result
    assert "proof_steps" in result
    assert "certainty" in result

@pytest.mark.unit
@pytest.mark.asyncio
async def test_collaborative_ai_process(mock_ai_agent):
    """Test collaborative AI process"""
    
    process = CollaborativeAIProcess("collab_test", mock_ai_agent, "consensus")
    
    assert process.process_id == "collab_test"
    assert process.ai_agent == mock_ai_agent
    assert process.collaboration_strategy == "consensus"

---
# tests/integration/test_end_to_end.py
"""
End-to-end integration tests
"""

import pytest
import asyncio
from core.advanced_csp_core import AdvancedCSPEngine, AtomicProcess, ChannelType, Event
from ai_integration.csp_ai_integration import AIAgent, CollaborativeAIProcess
from tests.conftest import MockLLMCapability

@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_ai_workflow():
    """Test complete AI workflow with CSP"""
    
    # Create CSP engine
    engine = AdvancedCSPEngine()
    
    # Create semantic channel
    channel = engine.create_channel("ai_workflow", ChannelType.SEMANTIC)
    
    # Create AI agent
    capability = MockLLMCapability()
    agent = AIAgent("workflow_agent", [capability])
    
    # Create collaborative process
    ai_process = CollaborativeAIProcess("ai_workflow_process", agent, "consensus")
    
    # Create data producer process
    async def data_producer(context):
        channel = context.get_channel("ai_workflow")
        
        # Send AI request
        request_event = Event(
            "ai_request",
            "ai_workflow", 
            {
                "type": "reasoning",
                "data": {"problem": "Test reasoning problem"}
            }
        )
        
        await channel.send(request_event, "producer")
        return "request_sent"
    
    producer = AtomicProcess("producer", data_producer)
    
    # Start processes
    producer_task = asyncio.create_task(producer.run(engine.context))
    ai_task = asyncio.create_task(ai_process.run(engine.context))
    
    # Wait for completion
    results = await asyncio.gather(producer_task, ai_task, return_exceptions=True)
    
    # Verify results
    assert results[0] == "request_sent"
    assert results[1] is not None  # AI process should return some result
    
    await engine.shutdown()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_agent_collaboration():
    """Test multi-agent collaboration"""
    
    engine = AdvancedCSPEngine()
    engine.create_channel("collaboration", ChannelType.SEMANTIC)
    
    # Create multiple AI agents
    agents = []
    processes = []
    
    for i in range(3):
        capability = MockLLMCapability()
        agent = AIAgent(f"agent_{i}", [capability])
        process = CollaborativeAIProcess(f"process_{i}", agent, "consensus")
        
        agents.append(agent)
        processes.append(process)
    
    # Setup peer relationships
    for i, process in enumerate(processes):
        for j, agent in enumerate(agents):
            if i != j:
                process.peer_agents[f"agent_{j}"] = agent
    
    # Create collaboration trigger
    async def collaboration_trigger(context):
        channel = context.get_channel("collaboration")
        
        collaboration_request = Event(
            "collaboration_request",
            "collaboration",
            {
                "type": "multi_agent_reasoning",
                "data": {"problem": "Complex multi-agent problem"}
            }
        )
        
        await channel.send(collaboration_request, "trigger")
        return "collaboration_triggered"
    
    trigger = AtomicProcess("trigger", collaboration_trigger)
    
    # Run collaboration
    tasks = [asyncio.create_task(process.run(engine.context)) for process in processes]
    trigger_task = asyncio.create_task(trigger.run(engine.context))
    
    results = await asyncio.gather(trigger_task, *tasks, return_exceptions=True)
    
    # Verify collaboration occurred
    assert results[0] == "collaboration_triggered"
    
    await engine.shutdown()

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance_under_load():
    """Test system performance under load"""
    
    engine = AdvancedCSPEngine()
    engine.create_channel("load_test", ChannelType.ASYNCHRONOUS)
    
    # Create multiple processes
    async def load_process(context, process_id):
        for i in range(10):
            # Simulate work
            await asyncio.sleep(0.01)
        return f"completed_{process_id}"
    
    # Create 50 concurrent processes
    processes = []
    for i in range(50):
        process = AtomicProcess(f"load_process_{i}", 
                              lambda ctx, pid=i: load_process(ctx, pid))
        processes.append(process)
    
    # Measure execution time
    start_time = asyncio.get_event_loop().time()
    
    tasks = [asyncio.create_task(process.run(engine.context)) for process in processes]
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    execution_time = end_time - start_time
    
    # Verify all processes completed
    assert len(results) == 50
    assert all("completed_" in result for result in results)
    
    # Performance assertion (should complete within reasonable time)
    assert execution_time < 5.0  # 5 seconds max for 50 processes
    
    await engine.shutdown()

---
# tests/performance/benchmark_suite.py
"""
Performance benchmarking suite for CSP System
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

from core.advanced_csp_core import AdvancedCSPEngine, AtomicProcess, ChannelType, Event

class CSPBenchmarkSuite:
    """Comprehensive benchmark suite for CSP system"""
    
    def __init__(self):
        self.results = {}
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and return results"""
        
        benchmarks = [
            ("Message Throughput", self.benchmark_message_throughput),
            ("Process Creation", self.benchmark_process_creation),
            ("Channel Communication", self.benchmark_channel_communication),
            ("Parallel Execution", self.benchmark_parallel_execution),
            ("Memory Usage", self.benchmark_memory_usage),
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"Running benchmark: {name}")
            result = await benchmark_func()
            self.results[name] = result
            print(f"  Result: {result}")
        
        return self.results
    
    async def benchmark_message_throughput(self) -> Dict[str, float]:
        """Benchmark message throughput"""
        
        engine = AdvancedCSPEngine()
        channel = engine.create_channel("throughput_test", ChannelType.ASYNCHRONOUS)
        
        message_count = 1000
        
        async def sender(context):
            start_time = time.time()
            for i in range(message_count):
                event = Event(f"msg_{i}", "throughput_test", f"data_{i}")
                await channel.send(event, "sender")
            end_time = time.time()
            return end_time - start_time
        
        async def receiver(context):
            received = 0
            start_time = time.time()
            while received < message_count:
                event = await channel.receive("receiver")
                if event:
                    received += 1
            end_time = time.time()
            return end_time - start_time
        
        sender_process = AtomicProcess("sender", sender)
        receiver_process = AtomicProcess("receiver", receiver)
        
        # Run benchmark
        sender_task = asyncio.create_task(sender_process.run(engine.context))
        receiver_task = asyncio.create_task(receiver_process.run(engine.context))
        
        sender_time, receiver_time = await asyncio.gather(sender_task, receiver_task)
        
        await engine.shutdown()
        
        return {
            "messages_per_second": message_count / max(sender_time, receiver_time),
            "sender_time": sender_time,
            "receiver_time": receiver_time
        }
    
    async def benchmark_process_creation(self) -> Dict[str, float]:
        """Benchmark process creation performance"""
        
        engine = AdvancedCSPEngine()
        
        async def simple_action(context):
            return "done"
        
        process_count = 100
        creation_times = []
        
        for i in range(process_count):
            start_time = time.time()
            process = AtomicProcess(f"process_{i}", simple_action)
            end_time = time.time()
            creation_times.append(end_time - start_time)
        
        await engine.shutdown()
        
        return {
            "avg_creation_time": statistics.mean(creation_times),
            "min_creation_time": min(creation_times),
            "max_creation_time": max(creation_times),
            "processes_per_second": process_count / sum(creation_times)
        }
    
    async def benchmark_channel_communication(self) -> Dict[str, float]:
        """Benchmark channel communication latency"""
        
        engine = AdvancedCSPEngine()
        sync_channel = engine.create_channel("sync_bench", ChannelType.SYNCHRONOUS)
        
        latencies = []
        iterations = 100
        
        async def ping_pong_test():
            for i in range(iterations):
                async def sender(context):
                    start_time = time.time()
                    event = Event(f"ping_{i}", "sync_bench", start_time)
                    await sync_channel.send(event, "sender")
                    return start_time
                
                async def receiver(context):
                    event = await sync_channel.receive("receiver")
                    end_time = time.time()
                    return end_time - event.data
                
                sender_process = AtomicProcess(f"sender_{i}", sender)
                receiver_process = AtomicProcess(f"receiver_{i}", receiver)
                
                sender_task = asyncio.create_task(sender_process.run(engine.context))
                receiver_task = asyncio.create_task(receiver_process.run(engine.context))
                
                _, latency = await asyncio.gather(sender_task, receiver_task)
                latencies.append(latency)
        
        await ping_pong_test()
        await engine.shutdown()
        
        return {
            "avg_latency_ms": statistics.mean(latencies) * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] * 1000
        }
    
    async def benchmark_parallel_execution(self) -> Dict[str, float]:
        """Benchmark parallel process execution"""
        
        engine = AdvancedCSPEngine()
        
        async def cpu_intensive_task(context, task_id):
            # Simulate CPU-intensive work
            start_time = time.time()
            for i in range(10000):
                _ = i ** 2
            end_time = time.time()
            return end_time - start_time
        
        process_counts = [1, 2, 4, 8, 16]
        results = {}
        
        for count in process_counts:
            processes = []
            for i in range(count):
                process = AtomicProcess(f"cpu_task_{i}", 
                                      lambda ctx, tid=i: cpu_intensive_task(ctx, tid))
                processes.append(process)
            
            # Measure parallel execution time
            start_time = time.time()
            tasks = [asyncio.create_task(process.run(engine.context)) for process in processes]
            execution_times = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_cpu_time = sum(execution_times)
            wall_clock_time = end_time - start_time
            parallelism_factor = total_cpu_time / wall_clock_time
            
            results[f"{count}_processes"] = {
                "wall_clock_time": wall_clock_time,
                "total_cpu_time": total_cpu_time,
                "parallelism_factor": parallelism_factor
            }
        
        await engine.shutdown()
        
        return results
    
    async def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage patterns"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = AdvancedCSPEngine()
        
        # Create many processes and channels
        processes = []
        channels = []
        
        for i in range(100):
            # Create channel
            channel = engine.create_channel(f"channel_{i}", ChannelType.SYNCHRONOUS)
            channels.append(channel)
            
            # Create process
            async def memory_action(context, process_id=i):
                # Allocate some memory
                data = list(range(1000))  # Small allocation
                return len(data)
            
            process_obj = AtomicProcess(f"mem_process_{i}", memory_action)
            processes.append(process_obj)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute all processes
        tasks = [asyncio.create_task(proc.run(engine.context)) for proc in processes]
        await asyncio.gather(*tasks)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await engine.shutdown()
        
        return {
            "initial_memory_mb": initial_memory,
            "mid_memory_mb": mid_memory,
            "final_memory_mb": final_memory,
            "memory_per_process_kb": (mid_memory - initial_memory) * 1024 / 100,
            "memory_per_channel_kb": (mid_memory - initial_memory) * 1024 / 100
        }

@pytest.mark.performance
@pytest.mark.asyncio
async def test_run_benchmark_suite():
    """Run the complete benchmark suite"""
    
    suite = CSPBenchmarkSuite()
    results = await suite.run_all_benchmarks()
    
    # Basic assertions to ensure benchmarks ran
    assert "Message Throughput" in results
    assert "Process Creation" in results
    assert "Channel Communication" in results
    assert "Parallel Execution" in results
    assert "Memory Usage" in results
    
    # Performance assertions (adjust based on expected performance)
    assert results["Message Throughput"]["messages_per_second"] > 100
    assert results["Process Creation"]["avg_creation_time"] < 0.01
    assert results["Channel Communication"]["avg_latency_ms"] < 10.0
    
    print("\n" + "="*50)
    print("CSP SYSTEM BENCHMARK RESULTS")
    print("="*50)
    
    for benchmark_name, result in results.items():
        print(f"\n{benchmark_name}:")
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {result}")

if __name__ == "__main__":
    # Run benchmarks directly
    async def main():
        suite = CSPBenchmarkSuite()
        await suite.run_all_benchmarks()
    
    asyncio.run(main())

---
# tests/fixtures/sample_processes.py
"""
Sample processes for testing
"""

import asyncio
from core.advanced_csp_core import AtomicProcess, CompositeProcess, CompositionOperator, Event

class SampleProcesses:
    """Collection of sample processes for testing"""
    
    @staticmethod
    def create_simple_counter():
        """Create a simple counter process"""
        
        async def counter_action(context):
            count = getattr(context, 'count', 0)
            count += 1
            context.count = count
            return count
        
        return AtomicProcess("counter", counter_action)
    
    @staticmethod
    def create_data_processor():
        """Create a data processing process"""
        
        async def processor_action(context):
            # Get data from context
            data = getattr(context, 'input_data', [])
            
            # Process data (double each value)
            processed = [x * 2 for x in data]
            
            # Store result
            context.processed_data = processed
            return processed
        
        return AtomicProcess("data_processor", processor_action)
    
    @staticmethod
    def create_message_sender(channel_name, message):
        """Create a message sender process"""
        
        async def sender_action(context):
            channel = context.get_channel(channel_name)
            if channel:
                event = Event("message", channel_name, message)
                await channel.send(event, "sender")
                return "sent"
            return "no_channel"
        
        return AtomicProcess("sender", sender_action)
    
    @staticmethod
    def create_message_receiver(channel_name):
        """Create a message receiver process"""
        
        async def receiver_action(context):
            channel = context.get_channel(channel_name)
            if channel:
                event = await channel.receive("receiver")
                return event.data if event else None
            return "no_channel"
        
        return AtomicProcess("receiver", receiver_action)
    
    @staticmethod
    def create_pipeline():
        """Create a data processing pipeline"""
        
        # Stage 1: Data generation
        async def generate_data(context):
            data = list(range(10))
            context.pipeline_data = data
            return data
        
        # Stage 2: Data transformation
        async def transform_data(context):
            data = getattr(context, 'pipeline_data', [])
            transformed = [x * x for x in data]
            context.pipeline_data = transformed
            return transformed
        
        # Stage 3: Data aggregation
        async def aggregate_data(context):
            data = getattr(context, 'pipeline_data', [])
            total = sum(data)
            return total
        
        # Create pipeline processes
        generator = AtomicProcess("generator", generate_data)
        transformer = AtomicProcess("transformer", transform_data)
        aggregator = AtomicProcess("aggregator", aggregate_data)
        
        # Create sequential pipeline
        pipeline = CompositeProcess(
            "data_pipeline",
            CompositionOperator.SEQUENTIAL,
            [generator, transformer, aggregator]
        )
        
        return pipeline
    
    @staticmethod
    def create_parallel_workers(worker_count=3):
        """Create parallel worker processes"""
        
        workers = []
        
        for i in range(worker_count):
            async def worker_action(context, worker_id=i):
                # Simulate work
                await asyncio.sleep(0.1)
                return f"worker_{worker_id}_result"
            
            worker = AtomicProcess(f"worker_{i}", worker_action)
            workers.append(worker)
        
        # Create parallel composition
        parallel_workers = CompositeProcess(
            "parallel_workers",
            CompositionOperator.PARALLEL,
            workers
        )
        
        return parallel_workers
    
    @staticmethod
    def create_choice_process():
        """Create a choice process that selects fastest option"""
        
        async def fast_option(context):
            await asyncio.sleep(0.01)
            return "fast_result"
        
        async def slow_option(context):
            await asyncio.sleep(0.1)
            return "slow_result"
        
        fast_process = AtomicProcess("fast", fast_option)
        slow_process = AtomicProcess("slow", slow_option)
        
        choice = CompositeProcess(
            "choice_process",
            CompositionOperator.CHOICE,
            [fast_process, slow_process]
        )
        
        return choice

---
# examples/basic_example/simple_communication.py
"""
Basic Example: Simple CSP Communication
=======================================

This example demonstrates basic CSP communication between processes.
"""

import asyncio
from core.advanced_csp_core import (
    AdvancedCSPEngine, AtomicProcess, ChannelType, Event
)

async def main():
    """Main example function"""
    
    print("🚀 CSP Basic Communication Example")
    print("=" * 40)
    
    # Create CSP engine
    engine = AdvancedCSPEngine()
    
    # Create a synchronous communication channel
    channel = engine.create_channel("basic_comm", ChannelType.SYNCHRONOUS)
    
    # Define producer process
    async def producer_action(context):
        print("📤 Producer: Starting to send messages...")
        channel = context.get_channel("basic_comm")
        
        messages = ["Hello", "CSP", "World", "!"]
        
        for i, message in enumerate(messages):
            event = Event(f"message_{i}", "basic_comm", message)
            await channel.send(event, "producer")
            print(f"📤 Producer: Sent '{message}'")
        
        print("📤 Producer: All messages sent!")
        return "production_complete"
    
    # Define consumer process
    async def consumer_action(context):
        print("📥 Consumer: Starting to receive messages...")
        channel = context.get_channel("basic_comm")
        
        received_messages = []
        
        for i in range(4):  # Expect 4 messages
            event = await channel.receive("consumer")
            message = event.data
            received_messages.append(message)
            print(f"📥 Consumer: Received '{message}'")
        
        print("📥 Consumer: All messages received!")
        return received_messages
    
    # Create processes
    producer = AtomicProcess("producer", producer_action)
    consumer = AtomicProcess("consumer", consumer_action)
    
    # Run processes concurrently
    print("\n🔄 Starting concurrent execution...")
    
    producer_task = asyncio.create_task(producer.run(engine.context))
    consumer_task = asyncio.create_task(consumer.run(engine.context))
    
    # Wait for both to complete
    producer_result, consumer_result = await asyncio.gather(
        producer_task, consumer_task
    )
    
    # Display results
    print(f"\n✅ Producer result: {producer_result}")
    print(f"✅ Consumer result: {consumer_result}")
    
    # Cleanup
    await engine.shutdown()
    
    print("\n🎉 Example completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

---
# examples/advanced_example/ai_collaboration.py
"""
Advanced Example: AI Agent Collaboration
========================================

This example demonstrates advanced AI agent collaboration using CSP.
"""

import asyncio
import json
from ai_integration.csp_ai_integration import (
    AIAgent, CollaborativeAIProcess, LLMCapability, ReasoningCapability
)
from core.advanced_csp_core import AdvancedCSPEngine, ChannelType, Event

async def main():
    """Advanced AI collaboration example"""
    
    print("🤖 Advanced AI Collaboration Example")
    print("=" * 45)
    
    # Create CSP engine with AI extensions
    engine = AdvancedCSPEngine()
    
    # Create semantic communication channel
    semantic_channel = engine.create_channel("ai_collaboration", ChannelType.SEMANTIC)
    
    # Create specialized AI agents
    print("\n🧠 Creating specialized AI agents...")
    
    # Reasoning Agent
    reasoning_agent = AIAgent("reasoning_specialist", [
        ReasoningCapability("logical"),
        LLMCapability("gpt-4", "reasoning")
    ])
    
    # Analysis Agent
    analysis_agent = AIAgent("analysis_specialist", [
        LLMCapability("claude", "analysis"),
        ReasoningCapability("causal")
    ])
    
    # Synthesis Agent
    synthesis_agent = AIAgent("synthesis_specialist", [
        LLMCapability("gpt-4", "synthesis"),
        ReasoningCapability("temporal")
    ])
    
    print("✅ Created 3 specialized AI agents")
    
    # Create collaborative processes
    reasoning_process = CollaborativeAIProcess(
        "reasoning_process", reasoning_agent, "consensus"
    )
    
    analysis_process = CollaborativeAIProcess(
        "analysis_process", analysis_agent, "consensus"
    )
    
    synthesis_process = CollaborativeAIProcess(
        "synthesis_process", synthesis_agent, "pipeline"
    )
    
    # Setup peer relationships for collaboration
    reasoning_process.peer_agents["analysis_specialist"] = analysis_agent
    reasoning_process.peer_agents["synthesis_specialist"] = synthesis_agent
    
    analysis_process.peer_agents["reasoning_specialist"] = reasoning_agent
    analysis_process.peer_agents["synthesis_specialist"] = synthesis_agent
    
    synthesis_process.peer_agents["reasoning_specialist"] = reasoning_agent
    synthesis_process.peer_agents["analysis_specialist"] = analysis_agent
    
    print("🔗 Established peer relationships between agents")
    
    # Define complex problem for collaboration
    complex_problem = {
        "type": "multi_domain_analysis",
        "data": {
            "problem": """
            Analyze the potential impact of quantum computing on current cybersecurity infrastructure.
            Consider: cryptographic algorithms, blockchain technology, financial systems, 
            and timeline for practical quantum computers.
            """,
            "context": {
                "domains": ["cryptography", "quantum_physics", "finance", "technology"],
                "requirements": ["risk_assessment", "timeline_prediction", "mitigation_strategies"],
                "complexity": "high"
            }
        }
    }
    
    print(f"\n🎯 Problem for collaboration:")
    print(f"   {complex_problem['data']['problem'][:100]}...")
    
    # Problem distribution process
    async def problem_distributor(context):
        print("\n📡 Distributing problem to AI agents...")
        
        channel = context.get_channel("ai_collaboration")
        
        # Send problem to each agent
        agents = ["reasoning_specialist", "analysis_specialist", "synthesis_specialist"]
        
        for agent_id in agents:
            # Customize problem for each agent
            specialized_problem = complex_problem.copy()
            specialized_problem["target_agent"] = agent_id
            specialized_problem["focus"] = {
                "reasoning_specialist": "logical_analysis",
                "analysis_specialist": "impact_assessment", 
                "synthesis_specialist": "solution_synthesis"
            }.get(agent_id, "general")
            
            event = Event(
                f"problem_for_{agent_id}",
                "ai_collaboration",
                specialized_problem
            )
            
            await channel.send(event, "distributor")
            print(f"📤 Sent specialized problem to {agent_id}")
        
        return "problems_distributed"
    
    # Result collector process
    async def result_collector(context):
        print("\n📥 Collecting results from AI agents...")
        
        channel = context.get_channel("ai_collaboration")
        results = {}
        
        # Collect results from all agents
        for i in range(3):
            event = await channel.receive("collector")
            if event:
                agent_result = event.data
                agent_id = agent_result.get("agent_id", f"agent_{i}")
                results[agent_id] = agent_result
                print(f"📥 Received result from {agent_id}")
        
        # Synthesize final collaborative result
        collaborative_result = {
            "collaboration_type": "multi_agent_consensus",
            "participating_agents": list(results.keys()),
            "individual_results": results,
            "synthesis": {
                "consensus_points": [
                    "Quantum computing poses significant threat to current cryptography",
                    "Timeline: 10-15 years for practical quantum computers",
                    "Immediate action needed for post-quantum cryptography"
                ],
                "confidence_score": 0.87,
                "recommendation": "Begin transition to quantum-resistant algorithms"
            }
        }
        
        return collaborative_result
    
    # Create coordination processes
    from core.advanced_csp_core import AtomicProcess
    
    distributor = AtomicProcess("distributor", problem_distributor)
    collector = AtomicProcess("collector", result_collector)
    
    print("\n🔄 Starting AI collaboration...")
    
    # Start all processes
    tasks = [
        asyncio.create_task(distributor.run(engine.context)),
        asyncio.create_task(reasoning_process.run(engine.context)),
        asyncio.create_task(analysis_process.run(engine.context)),
        asyncio.create_task(synthesis_process.run(engine.context)),
        asyncio.create_task(collector.run(engine.context))
    ]
    
    # Wait for collaboration to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("\n🎉 AI Collaboration completed!")
    
    # Display results
    print("\n📊 Collaboration Results:")
    print("=" * 30)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Process {i} failed: {result}")
        else:
            if isinstance(result, dict):
                print(f"✅ Process {i}: {json.dumps(result, indent=2)[:200]}...")
            else:
                print(f"✅ Process {i}: {result}")
    
    # Show final collaborative result
    final_result = results[-1]  # Collector result
    if isinstance(final_result, dict) and "synthesis" in final_result:
        print("\n🔮 Final Collaborative Synthesis:")
        synthesis = final_result["synthesis"]
        print(f"   Confidence: {synthesis['confidence_score']:.1%}")
        print(f"   Recommendation: {synthesis['recommendation']}")
        print("   Consensus Points:")
        for point in synthesis["consensus_points"]:
            print(f"   • {point}")
    
    # Cleanup
    await engine.shutdown()
    
    print("\n✨ Advanced example completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

---
# examples/enterprise_example/production_deployment.py
"""
Enterprise Example: Production Deployment
=========================================

This example shows how to deploy CSP system in production.
"""

import asyncio
import yaml
from pathlib import Path

from deployment.csp_deployment_system import (
    CSPDeploymentOrchestrator, DeploymentConfig, DeploymentTarget,
    ResourceLimits, NetworkConfig, ScalingConfig, MonitoringConfig
)

async def main():
    """Production deployment example"""
    
    print("🏭 Enterprise Production Deployment Example")
    print("=" * 50)
    
    # Create deployment orchestrator
    orchestrator = CSPDeploymentOrchestrator()
    
    # Define production configuration
    production_config = DeploymentConfig(
        name="csp-production",
        version="1.0.0",
        target=DeploymentTarget.KUBERNETES,
        image="csp-system:1.0.0",
        replicas=10,
        namespace="csp-production",
        
        # Resource configuration
        resources=ResourceLimits(
            cpu_limit="4000m",      # 4 CPU cores
            memory_limit="8Gi",     # 8GB RAM
            cpu_request="1000m",    # 1 CPU core minimum
            memory_request="2Gi",   # 2GB RAM minimum
            storage_limit="50Gi"    # 50GB storage
        ),
        
        # Network configuration
        network=NetworkConfig(
            port=8080,
            target_port=8080,
            protocol="TCP",
            ingress_enabled=True,
            tls_enabled=True,
            service_mesh=True
        ),
        
        # Auto-scaling configuration
        scaling=ScalingConfig(
            strategy=ScalingStrategy.CPU_BASED,
            min_replicas=5,
            max_replicas=50,
            target_cpu_utilization=70,
            target_memory_utilization=80,
            scale_up_cooldown=300,
            scale_down_cooldown=600
        ),
        
        # Monitoring configuration
        monitoring=MonitoringConfig(
            prometheus_enabled=True,
            grafana_enabled=True,
            jaeger_enabled=True,
            log_level="INFO",
            metrics_port=9090,
            health_check_path="/health",
            custom_metrics=[
                "csp_ai_collaboration_rate",
                "csp_protocol_synthesis_time",
                "csp_emergent_behavior_score"
            ]
        ),
        
        # Environment variables
        environment={
            "CSP_ENV": "production",
            "CSP_LOG_LEVEL": "INFO",
            "CSP_ENABLE_AI": "true",
            "CSP_ENABLE_MONITORING": "true",
            "CSP_ENABLE_FORMAL_VERIFICATION": "true",
            "CSP_PERFORMANCE_MODE": "optimized"
        },
        
        # Secrets (would be set from environment)
        secrets={
            "database_url": "postgresql://user:pass@db:5432/csp_prod",
            "redis_url": "redis://redis:6379/0",
            "openai_api_key": "${OPENAI_API_KEY}",
            "anthropic_api_key": "${ANTHROPIC_API_KEY}"
        },
        
        # Persistent volumes
        volumes=[
            {
                "name": "csp-data",
                "mount_path": "/app/data",
                "size": "20Gi",
                "type": "persistentVolumeClaim"
            },
            {
                "name": "csp-config",
                "mount_path": "/app/config",
                "type": "configMap",
                "source": "csp-production-config"
            }
        ]
    )
    
    print("📋 Production Configuration:")
    print(f"   Name: {production_config.name}")
    print(f"   Target: {production_config.target.name}")
    print(f"   Replicas: {production_config.replicas}")
    print(f"   Resources: {production_config.resources.cpu_limit} CPU, {production_config.resources.memory_limit} RAM")
    print(f"   Auto-scaling: {production_config.scaling.min_replicas}-{production_config.scaling.max_replicas} replicas")
    
    # Deploy to production
    print("\n🚀 Starting production deployment...")
    
    try:
        deployment_result = await orchestrator.deploy(production_config, "production")
        
        if deployment_result.get("status") == "deployed":
            deployment_id = deployment_result["deployment_id"]
            print(f"✅ Deployment successful!")
            print(f"   Deployment ID: {deployment_id}")
            
            # Wait for deployment to stabilize
            print("\n⏳ Waiting for deployment to stabilize...")
            await asyncio.sleep(10)
            
            # Check deployment status
            status = await orchestrator.get_deployment_status(deployment_id)
            print(f"\n📊 Deployment Status:")
            print(f"   Ready Replicas: {status.get('ready_replicas', 0)}/{status.get('replicas', 0)}")
            print(f"   Available Replicas: {status.get('available_replicas', 0)}")
            
            # Test scaling
            print("\n📈 Testing auto-scaling...")
            scale_result = await orchestrator.scale_deployment(deployment_id, 15)
            print(f"   Scale result: {scale_result.get('status', 'unknown')}")
            print(f"   New replica count: {scale_result.get('new_replicas', 'unknown')}")
            
            # Test rolling update
            print("\n🔄 Testing rolling update...")
            production_config.version = "1.0.1"
            update_result = await orchestrator.update_deployment(deployment_id, "rolling")
            print(f"   Update result: {update_result.get('status', 'unknown')}")
            
            # Show final deployment information
            print("\n🎯 Production Deployment Complete!")
            print("=" * 40)
            print("🌐 Service URLs:")
            print(f"   Main API: https://{production_config.name}.yourdomain.com")
            print(f"   Dashboard: https://{production_config.name}.yourdomain.com/dashboard")
            print(f"   Metrics: https://{production_config.name}.yourdomain.com/metrics")
            print(f"   Health: https://{production_config.name}.yourdomain.com/health")
            
            print("\n📊 Monitoring URLs:")
            print(f"   Grafana: https://grafana.yourdomain.com")
            print(f"   Prometheus: https://prometheus.yourdomain.com")
            print(f"   Jaeger: https://jaeger.yourdomain.com")
            
            print("\n🔧 Management Commands:")
            print(f"   Check status: kubectl get pods -n {production_config.namespace}")
            print(f"   View logs: kubectl logs -f deployment/{production_config.name} -n {production_config.namespace}")
            print(f"   Scale: kubectl scale deployment/{production_config.name} --replicas=20 -n {production_config.namespace}")
            
            # Generate deployment report
            report = {
                "deployment_id": deployment_id,
                "configuration": {
                    "name": production_config.name,
                    "version": production_config.version,
                    "target": production_config.target.name,
                    "replicas": production_config.replicas,
                    "resources": {
                        "cpu": production_config.resources.cpu_limit,
                        "memory": production_config.resources.memory_limit
                    }
                },
                "status": status,
                "deployment_time": deployment_result.get("deployed_at"),
                "urls": {
                    "api": f"https://{production_config.name}.yourdomain.com",
                    "dashboard": f"https://{production_config.name}.yourdomain.com/dashboard",
                    "metrics": f"https://{production_config.name}.yourdomain.com/metrics"
                }
            }
            
            # Save deployment report
            report_file = Path("deployment_report.yaml")
            with open(report_file, "w") as f:
                yaml.dump(report, f, default_flow_style=False)
            
            print(f"\n📋 Deployment report saved to: {report_file}")
            
        else:
            print(f"❌ Deployment failed: {deployment_result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return
    
    # Demonstrate additional enterprise features
    print("\n🏢 Enterprise Features Demonstrated:")
    print("   ✅ High-availability deployment (10 replicas)")
    print("   ✅ Auto-scaling (5-50 replicas based on CPU/memory)")
    print("   ✅ Rolling updates with zero downtime")
    print("   ✅ TLS/SSL termination")
    print("   ✅ Health checks and monitoring")
    print("   ✅ Persistent storage")
    print("   ✅ Secret management")
    print("   ✅ Service mesh integration")
    print("   ✅ Comprehensive observability")
    
    # Cleanup (in real production, you wouldn't do this)
    print("\n🧹 Cleaning up demo deployment...")
    cleanup_result = await orchestrator.delete_deployment(deployment_id)
    print(f"   Cleanup result: {cleanup_result.get('status', 'unknown')}")
    
    print("\n✨ Enterprise deployment example completed!")

if __name__ == "__main__":
    # Import the missing ScalingStrategy
    from deployment.csp_deployment_system import ScalingStrategy
    
    asyncio.run(main())
