# File: backend/ai/ai_integration.py
"""
AI Integration Components
========================
Advanced AI components for the CSP Visual Designer
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import os
from datetime import datetime, timedelta

import openai
from openai import AsyncOpenAI
import anthropic
from anthropic import AsyncAnthropic
import tiktoken
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from backend.components.registry import ComponentBase, ComponentMetadata, ComponentCategory, ComponentPort, PortType
from backend.database.connection import get_cache_manager

logger = logging.getLogger(__name__)

# ============================================================================
# AI PROVIDER CONFIGURATIONS
# ============================================================================

class AIProvider(str, Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class AIModelConfig:
    """Configuration for AI models"""
    provider: AIProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    
    # Model-specific settings
    context_window: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False
    cost_per_1k_tokens: float = 0.002

# Predefined model configurations
MODEL_CONFIGS = {
    "gpt-4": AIModelConfig(
        provider=AIProvider.OPENAI,
        model_name="gpt-4",
        context_window=8192,
        supports_function_calling=True,
        cost_per_1k_tokens=0.03
    ),
    "gpt-3.5-turbo": AIModelConfig(
        provider=AIProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        context_window=4096,
        supports_function_calling=True,
        cost_per_1k_tokens=0.002
    ),
    "claude-3-sonnet": AIModelConfig(
        provider=AIProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        context_window=200000,
        cost_per_1k_tokens=0.003
    ),
    "claude-3-haiku": AIModelConfig(
        provider=AIProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        context_window=200000,
        cost_per_1k_tokens=0.00025
    )
}

# ============================================================================
# AI CLIENT MANAGERS
# ============================================================================

class AIClientManager:
    """Manages AI client connections and rate limiting"""
    
    def __init__(self):
        self.clients: Dict[AIProvider, Any] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.token_usage: Dict[str, Dict[str, int]] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize AI clients"""
        if self._initialized:
            return
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.clients[AIProvider.OPENAI] = AsyncOpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized")
        
        # Initialize Anthropic client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.clients[AIProvider.ANTHROPIC] = AsyncAnthropic(api_key=anthropic_api_key)
            logger.info("✅ Anthropic client initialized")
        
        # Initialize HuggingFace transformers
        try:
            # This would be initialized with specific models as needed
            logger.info("✅ HuggingFace support available")
        except Exception as e:
            logger.warning(f"HuggingFace initialization failed: {e}")
        
        self._initialized = True
        logger.info("🤖 AI Client Manager initialized")
    
    def get_client(self, provider: AIProvider):
        """Get AI client for provider"""
        if not self._initialized:
            raise RuntimeError("AI Client Manager not initialized")
        
        return self.clients.get(provider)
    
    async def check_rate_limit(self, provider: AIProvider, model: str) -> bool:
        """Check if rate limit allows request"""
        key = f"{provider.value}:{model}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "requests": 0,
                "tokens": 0,
                "window_start": time.time()
            }
        
        rate_limit = self.rate_limits[key]
        current_time = time.time()
        
        # Reset window if needed (1 minute windows)
        if current_time - rate_limit["window_start"] > 60:
            rate_limit["requests"] = 0
            rate_limit["tokens"] = 0
            rate_limit["window_start"] = current_time
        
        # Check limits (simplified - would be model-specific in production)
        max_requests_per_minute = 60
        max_tokens_per_minute = 10000
        
        if (rate_limit["requests"] >= max_requests_per_minute or 
            rate_limit["tokens"] >= max_tokens_per_minute):
            return False
        
        return True
    
    async def record_usage(self, provider: AIProvider, model: str, tokens: int):
        """Record token usage for rate limiting and billing"""
        key = f"{provider.value}:{model}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "requests": 0,
                "tokens": 0,
                "window_start": time.time()
            }
        
        self.rate_limits[key]["requests"] += 1
        self.rate_limits[key]["tokens"] += tokens
        
        # Track total usage
        if key not in self.token_usage:
            self.token_usage[key] = {"total_tokens": 0, "total_requests": 0}
        
        self.token_usage[key]["total_tokens"] += tokens
        self.token_usage[key]["total_requests"] += 1

# Global AI client manager
ai_client_manager = AIClientManager()

# ============================================================================
# AI SERVICE IMPLEMENTATIONS
# ============================================================================

class BaseAIService(ABC):
    """Base class for AI services"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.client_manager = ai_client_manager
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], 
                              **kwargs) -> Dict[str, Any]:
        """Generate AI response"""
        pass
    
    @abstractmethod
    async def generate_streaming_response(self, messages: List[Dict[str, str]], 
                                        **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming AI response"""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            if self.config.provider == AIProvider.OPENAI:
                encoding = tiktoken.encoding_for_model(self.config.model_name)
                return len(encoding.encode(text))
            else:
                # Rough estimation for other providers
                return len(text.split()) * 1.3
        except Exception:
            return len(text.split()) * 1.3

class OpenAIService(BaseAIService):
    """OpenAI API service implementation"""
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        client = self.client_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        # Check rate limits
        if not await self.client_manager.check_rate_limit(
            AIProvider.OPENAI, self.config.model_name
        ):
            raise RuntimeError("Rate limit exceeded")
        
        try:
            response = await client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
                presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
                timeout=self.config.timeout
            )
            
            # Record usage
            await self.client_manager.record_usage(
                AIProvider.OPENAI,
                self.config.model_name,
                response.usage.total_tokens
            )
            
            return {
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    async def generate_streaming_response(self, messages: List[Dict[str, str]], 
                                        **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using OpenAI API"""
        client = self.client_manager.get_client(AIProvider.OPENAI)
        if not client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            stream = await client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                stream=True,
                timeout=self.config.timeout
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise RuntimeError(f"OpenAI streaming error: {str(e)}")

class AnthropicService(BaseAIService):
    """Anthropic API service implementation"""
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              **kwargs) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        client = self.client_manager.get_client(AIProvider.ANTHROPIC)
        if not client:
            raise RuntimeError("Anthropic client not available")
        
        # Check rate limits
        if not await self.client_manager.check_rate_limit(
            AIProvider.ANTHROPIC, self.config.model_name
        ):
            raise RuntimeError("Rate limit exceeded")
        
        try:
            # Convert messages to Anthropic format
            system_message = ""
            conversation_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append(msg)
            
            response = await client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                system=system_message if system_message else None,
                messages=conversation_messages
            )
            
            # Estimate token usage (Anthropic doesn't always provide exact counts)
            estimated_tokens = self.count_tokens(
                system_message + " ".join([m["content"] for m in conversation_messages])
            )
            
            # Record usage
            await self.client_manager.record_usage(
                AIProvider.ANTHROPIC,
                self.config.model_name,
                estimated_tokens
            )
            
            return {
                "content": response.content[0].text,
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', estimated_tokens // 2),
                    "completion_tokens": getattr(response.usage, 'output_tokens', estimated_tokens // 2),
                    "total_tokens": estimated_tokens
                },
                "model": response.model,
                "finish_reason": response.stop_reason
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    async def generate_streaming_response(self, messages: List[Dict[str, str]], 
                                        **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Anthropic API"""
        client = self.client_manager.get_client(AIProvider.ANTHROPIC)
        if not client:
            raise RuntimeError("Anthropic client not available")
        
        try:
            # Convert messages
            system_message = ""
            conversation_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append(msg)
            
            async with client.messages.stream(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                system=system_message if system_message else None,
                messages=conversation_messages
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise RuntimeError(f"Anthropic streaming error: {str(e)}")

# ============================================================================
# AI COMPONENT IMPLEMENTATIONS
# ============================================================================

class AdvancedAIAgentComponent(ComponentBase):
    """Advanced AI Agent with multi-model support and conversation memory"""
    
    metadata = ComponentMetadata(
        component_type="advanced_ai_agent",
        category=ComponentCategory.AI,
        display_name="Advanced AI Agent",
        description="Multi-model AI agent with conversation memory and function calling",
        icon="robot",
        color="#FF6B6B",
        input_ports=[
            ComponentPort("input", PortType.TEXT, required=True, description="User input"),
            ComponentPort("system_prompt", PortType.TEXT, required=False, description="System prompt"),
            ComponentPort("memory_context", PortType.JSON, required=False, description="Conversation memory"),
            ComponentPort("functions", PortType.JSON, required=False, description="Available functions"),
        ],
        output_ports=[
            ComponentPort("response", PortType.TEXT, description="AI response"),
            ComponentPort("usage_stats", PortType.JSON, description="Token usage and costs"),
            ComponentPort("updated_memory", PortType.JSON, description="Updated conversation memory"),
            ComponentPort("function_calls", PortType.JSON, description="Function calls made"),
        ],
        default_properties={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "memory_enabled": True,
            "max_memory_turns": 10,
            "system_prompt": "You are a helpful AI assistant specialized in CSP process design."
        }
    )
    
    def __init__(self, node_id: str, properties: Dict[str, Any] = None):
        super().__init__(node_id, properties)
        self.ai_service: Optional[BaseAIService] = None
        self.conversation_memory: List[Dict[str, str]] = []
        self.total_usage: Dict[str, int] = {"tokens": 0, "requests": 0}
    
    async def initialize(self) -> bool:
        """Initialize AI agent with selected model"""
        try:
            model_name = self.properties.get("model", "gpt-4")
            
            if model_name not in MODEL_CONFIGS:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            config = MODEL_CONFIGS[model_name]
            
            # Initialize AI client manager if needed
            if not ai_client_manager._initialized:
                await ai_client_manager.initialize()
            
            # Create appropriate service
            if config.provider == AIProvider.OPENAI:
                self.ai_service = OpenAIService(config)
            elif config.provider == AIProvider.ANTHROPIC:
                self.ai_service = AnthropicService(config)
            else:
                logger.error(f"Unsupported provider: {config.provider}")
                return False
            
            logger.info(f"Advanced AI Agent {self.node_id} initialized with {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Advanced AI Agent {self.node_id}: {e}")
            return False
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI processing with memory and function calling"""
        try:
            if not self.ai_service:
                return {"error": "AI service not initialized"}
            
            user_input = inputs.get("input", "")
            system_prompt = inputs.get("system_prompt", self.properties.get("system_prompt"))
            memory_context = inputs.get("memory_context", {})
            available_functions = inputs.get("functions", [])
            
            if not user_input:
                return {"error": "No input provided"}
            
            # Build conversation context
            messages = []
            
            # Add system prompt
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add memory context if enabled
            if self.properties.get("memory_enabled", True):
                # Load previous memory
                if memory_context.get("conversation_history"):
                    self.conversation_memory = memory_context["conversation_history"]
                
                # Add recent conversation history
                max_turns = self.properties.get("max_memory_turns", 10)
                recent_memory = self.conversation_memory[-max_turns:]
                messages.extend(recent_memory)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Generate AI response
            start_time = time.time()
            response_data = await self.ai_service.generate_response(
                messages,
                max_tokens=self.properties.get("max_tokens", 1000),
                temperature=self.properties.get("temperature", 0.7)
            )
            execution_time = time.time() - start_time
            
            ai_response = response_data["content"]
            usage = response_data["usage"]
            
            # Update conversation memory
            if self.properties.get("memory_enabled", True):
                self.conversation_memory.append({"role": "user", "content": user_input})
                self.conversation_memory.append({"role": "assistant", "content": ai_response})
                
                # Trim memory if too long
                max_turns = self.properties.get("max_memory_turns", 10) * 2  # *2 for user+assistant pairs
                if len(self.conversation_memory) > max_turns:
                    self.conversation_memory = self.conversation_memory[-max_turns:]
            
            # Update total usage
            self.total_usage["tokens"] += usage["total_tokens"]
            self.total_usage["requests"] += 1
            
            # Calculate costs
            model_config = MODEL_CONFIGS[self.properties.get("model", "gpt-4")]
            estimated_cost = (usage["total_tokens"] / 1000) * model_config.cost_per_1k_tokens
            
            return {
                "response": ai_response,
                "usage_stats": {
                    "current_request": usage,
                    "session_total": self.total_usage.copy(),
                    "estimated_cost": round(estimated_cost, 6),
                    "execution_time": round(execution_time, 3),
                    "model": response_data["model"]
                },
                "updated_memory": {
                    "conversation_history": self.conversation_memory.copy()
                },
                "function_calls": []  # Would implement function calling logic here
            }
            
        except Exception as e:
            logger.error(f"Advanced AI Agent {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup AI agent resources"""
        self.conversation_memory.clear()
        self.ai_service = None

class AITextAnalyzerComponent(ComponentBase):
    """AI-powered text analysis component"""
    
    metadata = ComponentMetadata(
        component_type="ai_text_analyzer",
        category=ComponentCategory.AI,
        display_name="AI Text Analyzer",
        description="Analyze text for sentiment, entities, topics, and more",
        icon="analyze",
        color="#4ECDC4",
        input_ports=[
            ComponentPort("text", PortType.TEXT, required=True, description="Text to analyze"),
            ComponentPort("analysis_types", PortType.JSON, required=False, description="Types of analysis to perform"),
        ],
        output_ports=[
            ComponentPort("sentiment", PortType.JSON, description="Sentiment analysis results"),
            ComponentPort("entities", PortType.JSON, description="Named entity recognition"),
            ComponentPort("topics", PortType.JSON, description="Topic classification"),
            ComponentPort("summary", PortType.TEXT, description="Text summary"),
            ComponentPort("analysis_metadata", PortType.JSON, description="Analysis metadata"),
        ],
        default_properties={
            "model": "gpt-3.5-turbo",
            "analysis_types": ["sentiment", "entities", "topics", "summary"],
            "max_summary_length": 100
        }
    )
    
    def __init__(self, node_id: str, properties: Dict[str, Any] = None):
        super().__init__(node_id, properties)
        self.ai_service: Optional[BaseAIService] = None
    
    async def initialize(self) -> bool:
        """Initialize text analyzer"""
        try:
            model_name = self.properties.get("model", "gpt-3.5-turbo")
            config = MODEL_CONFIGS.get(model_name)
            
            if not config:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            if not ai_client_manager._initialized:
                await ai_client_manager.initialize()
            
            if config.provider == AIProvider.OPENAI:
                self.ai_service = OpenAIService(config)
            elif config.provider == AIProvider.ANTHROPIC:
                self.ai_service = AnthropicService(config)
            
            logger.info(f"AI Text Analyzer {self.node_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Text Analyzer {self.node_id}: {e}")
            return False
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text analysis"""
        try:
            if not self.ai_service:
                return {"error": "AI service not initialized"}
            
            text = inputs.get("text", "")
            analysis_types = inputs.get("analysis_types", self.properties.get("analysis_types", []))
            
            if not text:
                return {"error": "No text provided"}
            
            results = {}
            
            # Perform different types of analysis
            for analysis_type in analysis_types:
                if analysis_type == "sentiment":
                    results["sentiment"] = await self._analyze_sentiment(text)
                elif analysis_type == "entities":
                    results["entities"] = await self._extract_entities(text)
                elif analysis_type == "topics":
                    results["topics"] = await self._classify_topics(text)
                elif analysis_type == "summary":
                    results["summary"] = await self._generate_summary(text)
            
            # Add metadata
            results["analysis_metadata"] = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "analysis_types": analysis_types,
                "timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"AI Text Analyzer {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        prompt = f"""
        Analyze the sentiment of the following text. Provide a JSON response with:
        - sentiment: positive, negative, or neutral
        - confidence: float between 0 and 1
        - explanation: brief explanation of the sentiment

        Text: {text}

        Response format: {{"sentiment": "positive", "confidence": 0.9, "explanation": "..."}}
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.1)
        
        try:
            return json.loads(response["content"])
        except:
            return {"sentiment": "neutral", "confidence": 0.5, "explanation": "Could not parse sentiment"}
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        prompt = f"""
        Extract named entities from the following text. Provide a JSON array with entities:
        Each entity should have: name, type (PERSON, ORGANIZATION, LOCATION, etc.), start_pos, end_pos

        Text: {text}

        Response format: [{{"name": "Entity Name", "type": "PERSON", "start_pos": 0, "end_pos": 10}}]
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.1)
        
        try:
            return json.loads(response["content"])
        except:
            return []
    
    async def _classify_topics(self, text: str) -> List[Dict[str, Any]]:
        """Classify topics in text"""
        prompt = f"""
        Identify the main topics in the following text. Provide a JSON array with topics:
        Each topic should have: name, confidence (0-1), description

        Text: {text}

        Response format: [{{"name": "Topic Name", "confidence": 0.8, "description": "..."}}]
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.2)
        
        try:
            return json.loads(response["content"])
        except:
            return []
    
    async def _generate_summary(self, text: str) -> str:
        """Generate summary of text"""
        max_length = self.properties.get("max_summary_length", 100)
        
        prompt = f"""
        Summarize the following text in approximately {max_length} words or less:

        Text: {text}

        Summary:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.3)
        
        return response["content"].strip()
    
    async def cleanup(self):
        """Cleanup text analyzer resources"""
        self.ai_service = None

class AICodeGeneratorComponent(ComponentBase):
    """AI-powered code generation component"""
    
    metadata = ComponentMetadata(
        component_type="ai_code_generator",
        category=ComponentCategory.AI,
        display_name="AI Code Generator",
        description="Generate code from natural language descriptions",
        icon="code",
        color="#A8E6CF",
        input_ports=[
            ComponentPort("description", PortType.TEXT, required=True, description="Natural language description"),
            ComponentPort("language", PortType.TEXT, required=False, description="Programming language"),
            ComponentPort("context", PortType.JSON, required=False, description="Additional context or examples"),
        ],
        output_ports=[
            ComponentPort("generated_code", PortType.TEXT, description="Generated code"),
            ComponentPort("explanation", PortType.TEXT, description="Code explanation"),
            ComponentPort("tests", PortType.TEXT, description="Generated unit tests"),
            ComponentPort("documentation", PortType.TEXT, description="Code documentation"),
        ],
        default_properties={
            "model": "gpt-4",
            "language": "python",
            "include_tests": True,
            "include_documentation": True,
            "code_style": "clean",
            "max_code_tokens": 2000
        }
    )
    
    def __init__(self, node_id: str, properties: Dict[str, Any] = None):
        super().__init__(node_id, properties)
        self.ai_service: Optional[BaseAIService] = None
    
    async def initialize(self) -> bool:
        """Initialize code generator"""
        try:
            model_name = self.properties.get("model", "gpt-4")
            config = MODEL_CONFIGS.get(model_name)
            
            if not config:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            if not ai_client_manager._initialized:
                await ai_client_manager.initialize()
            
            if config.provider == AIProvider.OPENAI:
                self.ai_service = OpenAIService(config)
            elif config.provider == AIProvider.ANTHROPIC:
                self.ai_service = AnthropicService(config)
            
            logger.info(f"AI Code Generator {self.node_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Code Generator {self.node_id}: {e}")
            return False
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation"""
        try:
            if not self.ai_service:
                return {"error": "AI service not initialized"}
            
            description = inputs.get("description", "")
            language = inputs.get("language", self.properties.get("language", "python"))
            context = inputs.get("context", {})
            
            if not description:
                return {"error": "No description provided"}
            
            # Generate code
            generated_code = await self._generate_code(description, language, context)
            
            results = {"generated_code": generated_code}
            
            # Generate explanation
            results["explanation"] = await self._generate_explanation(generated_code, description)
            
            # Generate tests if requested
            if self.properties.get("include_tests", True):
                results["tests"] = await self._generate_tests(generated_code, language)
            
            # Generate documentation if requested
            if self.properties.get("include_documentation", True):
                results["documentation"] = await self._generate_documentation(generated_code, description)
            
            return results
            
        except Exception as e:
            logger.error(f"AI Code Generator {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def _generate_code(self, description: str, language: str, context: Dict[str, Any]) -> str:
        """Generate code from description"""
        context_str = ""
        if context:
            context_str = f"\nAdditional context: {json.dumps(context, indent=2)}"
        
        prompt = f"""
        Generate {language} code based on the following description:

        Description: {description}{context_str}

        Requirements:
        - Write clean, readable, and efficient code
        - Follow {language} best practices and conventions
        - Include appropriate error handling
        - Add inline comments where helpful
        - Make the code production-ready

        Generated {language} code:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(
            messages, 
            temperature=0.2,
            max_tokens=self.properties.get("max_code_tokens", 2000)
        )
        
        return response["content"].strip()
    
    async def _generate_explanation(self, code: str, description: str) -> str:
        """Generate explanation of the generated code"""
        prompt = f"""
        Explain the following code that was generated for: "{description}"

        Code:
        ```
        {code}
        ```

        Provide a clear explanation of:
        1. What the code does
        2. How it works
        3. Key components and their purposes
        4. Any important design decisions

        Explanation:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.3)
        
        return response["content"].strip()
    
    async def _generate_tests(self, code: str, language: str) -> str:
        """Generate unit tests for the code"""
        prompt = f"""
        Generate comprehensive unit tests for the following {language} code:

        Code:
        ```
        {code}
        ```

        Requirements for tests:
        - Test all major functions and methods
        - Include edge cases and error conditions
        - Use appropriate testing framework for {language}
        - Include setup and teardown if needed
        - Add descriptive test names and comments

        Generated tests:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.2)
        
        return response["content"].strip()
    
    async def _generate_documentation(self, code: str, description: str) -> str:
        """Generate documentation for the code"""
        prompt = f"""
        Generate comprehensive documentation for the following code:

        Original requirement: {description}

        Code:
        ```
        {code}
        ```

        Include:
        1. Overview and purpose
        2. Function/method documentation
        3. Parameter descriptions
        4. Return value descriptions
        5. Usage examples
        6. Any important notes or limitations

        Documentation:
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.ai_service.generate_response(messages, temperature=0.3)
        
        return response["content"].strip()
    
    async def cleanup(self):
        """Cleanup code generator resources"""
        self.ai_service = None

# ============================================================================
# AI UTILITY FUNCTIONS
# ============================================================================

async def get_ai_client_manager() -> AIClientManager:
    """Get the global AI client manager"""
    if not ai_client_manager._initialized:
        await ai_client_manager.initialize()
    return ai_client_manager

async def get_usage_statistics() -> Dict[str, Any]:
    """Get AI usage statistics"""
    manager = await get_ai_client_manager()
    
    total_tokens = sum(usage["total_tokens"] for usage in manager.token_usage.values())
    total_requests = sum(usage["total_requests"] for usage in manager.token_usage.values())
    
    # Calculate estimated costs
    total_cost = 0.0
    for key, usage in manager.token_usage.items():
        provider, model = key.split(":", 1)
        if model in MODEL_CONFIGS:
            cost_per_token = MODEL_CONFIGS[model].cost_per_1k_tokens / 1000
            total_cost += usage["total_tokens"] * cost_per_token
    
    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "estimated_total_cost": round(total_cost, 6),
        "usage_by_model": manager.token_usage,
        "rate_limits": manager.rate_limits
    }

async def reset_usage_statistics():
    """Reset AI usage statistics"""
    manager = await get_ai_client_manager()
    manager.token_usage.clear()
    manager.rate_limits.clear()
    logger.info("AI usage statistics reset")

# Register AI components with the component registry
async def register_ai_components():
    """Register AI components with the component registry"""
    from backend.components.registry import component_registry
    
    ai_components = [
        (AdvancedAIAgentComponent.metadata, AdvancedAIAgentComponent),
        (AITextAnalyzerComponent.metadata, AITextAnalyzerComponent),
        (AICodeGeneratorComponent.metadata, AICodeGeneratorComponent),
    ]
    
    for metadata, component_class in ai_components:
        await component_registry.register_component(metadata, component_class)
    
    logger.info("✅ AI components registered")
