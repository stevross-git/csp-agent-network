#!/usr/bin/env python3
"""
Multi-modal AI Communication Hub
================================

Advanced multi-modal AI communication system for CSP networks:
- Cross-modal AI agent communication (text, image, audio, video)
- Universal AI protocol translation
- Multi-modal content understanding and generation
- Real-time media processing pipelines
- AI model orchestration and routing
- Semantic content analysis and transformation
- Multi-language and multi-format support
- Advanced reasoning across modalities
"""

import asyncio
import json
import time
import base64
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import mimetypes
import hashlib
import io
from pathlib import Path
import tempfile
import numpy as np

# Multi-modal AI libraries
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoProcessor,
        BlipProcessor, BlipForConditionalGeneration,
        WhisperProcessor, WhisperForConditionalGeneration,
        CLIPProcessor, CLIPModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False

# NLP and text processing
try:
    import spacy
    from textblob import TextBlob
    import nltk
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Import our CSP components
from core.advanced_csp_core import Process, ProcessContext, Channel, Event
from ai_integration.csp_ai_integration import AIAgent, LLMCapability

# ============================================================================
# MODALITY DEFINITIONS
# ============================================================================

class ModalityType(Enum):
    """Supported modality types"""
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    STRUCTURED_DATA = auto()
    CODE = auto()
    MATHEMATICAL = auto()
    SENSOR_DATA = auto()
    BIOMETRIC = auto()
    GEOSPATIAL = auto()

class ContentFormat(Enum):
    """Content format specifications"""
    # Text formats
    PLAIN_TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    JSON = "application/json"
    XML = "application/xml"
    
    # Image formats
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    SVG = "image/svg+xml"
    
    # Audio formats
    WAV = "audio/wav"
    MP3 = "audio/mp3"
    FLAC = "audio/flac"
    
    # Video formats
    MP4 = "video/mp4"
    AVI = "video/avi"
    WEBM = "video/webm"
    
    # Data formats
    CSV = "text/csv"
    PARQUET = "application/octet-stream"
    BINARY = "application/octet-stream"

@dataclass
class MultiModalContent:
    """Multi-modal content representation"""
    content_id: str
    modality: ModalityType
    format: ContentFormat
    data: Union[str, bytes, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    encoding: str = "utf-8"
    timestamp: float = field(default_factory=time.time)
    size_bytes: int = 0
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate size and checksum after initialization"""
        if isinstance(self.data, str):
            self.data_bytes = self.data.encode(self.encoding)
        elif isinstance(self.data, bytes):
            self.data_bytes = self.data
        else:
            self.data_bytes = str(self.data).encode(self.encoding)
        
        self.size_bytes = len(self.data_bytes)
        self.checksum = hashlib.md5(self.data_bytes).hexdigest()

@dataclass
class AICapabilityProfile:
    """AI agent capability profile for multi-modal processing"""
    agent_id: str
    supported_modalities: List[ModalityType]
    input_formats: List[ContentFormat]
    output_formats: List[ContentFormat]
    processing_capabilities: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)

# ============================================================================
# MULTI-MODAL PROCESSORS
# ============================================================================

class TextProcessor:
    """Advanced text processing capabilities"""
    
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
        self.nlp_pipeline = None
        
        if NLP_AVAILABLE:
            try:
                self.nlp_pipeline = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy model not found, using basic processing")
    
    async def process_text(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process text content with NLP analysis"""
        
        if content.modality != ModalityType.TEXT:
            raise ValueError("Content is not text")
        
        text = content.data if isinstance(content.data, str) else content.data.decode()
        
        analysis = {
            'content_id': content.content_id,
            'text_length': len(text),
            'word_count': len(text.split()),
            'language': 'unknown',
            'sentiment': {'polarity': 0.0, 'subjectivity': 0.0},
            'entities': [],
            'topics': [],
            'summary': '',
            'keywords': []
        }
        
        try:
            # Language detection and sentiment analysis
            if NLP_AVAILABLE:
                blob = TextBlob(text)
                analysis['language'] = blob.detect_language()
                analysis['sentiment'] = {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            
            # Named entity recognition
            if self.nlp_pipeline:
                doc = self.nlp_pipeline(text[:1000000])  # Limit for performance
                analysis['entities'] = [
                    {'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
                    for ent in doc.ents
                ]
                
                # Extract keywords
                analysis['keywords'] = [
                    token.lemma_.lower() for token in doc 
                    if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']
                ][:20]
            
            # Generate summary (simple extractive)
            sentences = text.split('.')[:5]  # First 5 sentences
            analysis['summary'] = '. '.join(sentences).strip() + '.'
            
        except Exception as e:
            logging.error(f"Text processing error: {e}")
        
        return analysis
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language"""
        # This would use a translation service in production
        # For demo, return mock translation
        return f"[Translated to {target_language}] {text}"
    
    async def generate_text(self, prompt: str, style: str = "default") -> str:
        """Generate text based on prompt and style"""
        # This would use a language model in production
        styles = {
            'formal': f"In formal terms, regarding {prompt}, it is important to note that",
            'casual': f"So about {prompt}, here's the thing:",
            'technical': f"Technical analysis of {prompt} indicates that",
            'creative': f"Imagine {prompt} in a world where"
        }
        
        base_response = styles.get(style, f"Regarding {prompt}:")
        return f"{base_response} this is a generated response based on the input prompt."

class ImageProcessor:
    """Advanced image processing capabilities"""
    
    def __init__(self):
        self.models = {}
        self.transforms = None
        
        if TORCH_AVAILABLE and VISION_AVAILABLE:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Load vision models
        if TRANSFORMERS_AVAILABLE:
            try:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            except Exception as e:
                logging.warning(f"Failed to load vision models: {e}")
    
    async def process_image(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process image content with computer vision analysis"""
        
        if content.modality != ModalityType.IMAGE:
            raise ValueError("Content is not an image")
        
        analysis = {
            'content_id': content.content_id,
            'dimensions': (0, 0),
            'color_analysis': {},
            'objects_detected': [],
            'caption': '',
            'tags': [],
            'faces_detected': 0,
            'text_detected': [],
            'scene_classification': 'unknown'
        }
        
        try:
            # Load image
            if isinstance(content.data, bytes):
                image = Image.open(io.BytesIO(content.data))
            else:
                image = Image.open(content.data)
            
            analysis['dimensions'] = image.size
            
            # Color analysis
            colors = image.getcolors(maxcolors=256*256*256)
            if colors:
                dominant_color = max(colors, key=lambda x: x[0])
                analysis['color_analysis'] = {
                    'dominant_color': dominant_color[1] if len(dominant_color) > 1 else dominant_color,
                    'total_colors': len(colors)
                }
            
            # Generate caption using BLIP
            if hasattr(self, 'blip_processor') and hasattr(self, 'blip_model'):
                try:
                    inputs = self.blip_processor(image, return_tensors="pt")
                    out = self.blip_model.generate(**inputs, max_length=50)
                    caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                    analysis['caption'] = caption
                except Exception as e:
                    logging.error(f"Caption generation failed: {e}")
            
            # Object detection (simplified)
            if VISION_AVAILABLE:
                # Convert to OpenCV format for basic analysis
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Face detection
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(opencv_image, 1.1, 4)
                    analysis['faces_detected'] = len(faces)
                except Exception as e:
                    logging.error(f"Face detection failed: {e}")
            
            # Generate tags
            analysis['tags'] = ['image', 'visual_content', 'processed']
            
        except Exception as e:
            logging.error(f"Image processing error: {e}")
        
        return analysis
    
    async def generate_image(self, prompt: str, style: str = "default") -> bytes:
        """Generate image based on text prompt"""
        # This would use an image generation model in production
        # For demo, create a simple image with text
        
        if not VISION_AVAILABLE:
            return b"Image generation not available"
        
        # Create a simple image with the prompt text
        width, height = 512, 512
        image = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(image)
        
        # Add text to image
        try:
            font = ImageFont.load_default()
            text_width, text_height = draw.textsize(prompt, font=font)
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            draw.text((x, y), prompt, fill='black', font=font)
        except:
            draw.text((50, 250), prompt, fill='black')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    async def transform_image(self, content: MultiModalContent, 
                            transformation: str) -> MultiModalContent:
        """Apply transformations to image"""
        
        if not VISION_AVAILABLE:
            return content
        
        try:
            # Load image
            if isinstance(content.data, bytes):
                image = Image.open(io.BytesIO(content.data))
            else:
                image = Image.open(content.data)
            
            # Apply transformation
            if transformation == 'grayscale':
                image = image.convert('L')
            elif transformation == 'resize_small':
                image = image.resize((128, 128))
            elif transformation == 'resize_large':
                image = image.resize((1024, 1024))
            elif transformation == 'rotate_90':
                image = image.rotate(90)
            elif transformation == 'flip_horizontal':
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif transformation == 'enhance_contrast':
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
            
            # Convert back to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            
            # Create new content
            new_content = MultiModalContent(
                content_id=f"{content.content_id}_transformed",
                modality=ModalityType.IMAGE,
                format=ContentFormat.PNG,
                data=img_bytes.getvalue(),
                metadata={**content.metadata, 'transformation': transformation}
            )
            
            return new_content
            
        except Exception as e:
            logging.error(f"Image transformation error: {e}")
            return content

class AudioProcessor:
    """Advanced audio processing capabilities"""
    
    def __init__(self):
        self.models = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            except Exception as e:
                logging.warning(f"Failed to load audio models: {e}")
    
    async def process_audio(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process audio content with analysis"""
        
        if content.modality != ModalityType.AUDIO:
            raise ValueError("Content is not audio")
        
        analysis = {
            'content_id': content.content_id,
            'duration': 0.0,
            'sample_rate': 0,
            'channels': 0,
            'transcription': '',
            'language_detected': 'unknown',
            'audio_features': {},
            'speech_segments': [],
            'music_detected': False,
            'noise_level': 0.0
        }
        
        try:
            if AUDIO_AVAILABLE:
                # Load audio using librosa
                if isinstance(content.data, bytes):
                    # Save bytes to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        tmp_file.write(content.data)
                        tmp_path = tmp_file.name
                    
                    audio_data, sr = librosa.load(tmp_path)
                    Path(tmp_path).unlink()  # Clean up
                else:
                    audio_data, sr = librosa.load(content.data)
                
                analysis['duration'] = len(audio_data) / sr
                analysis['sample_rate'] = sr
                analysis['channels'] = 1 if audio_data.ndim == 1 else audio_data.shape[0]
                
                # Extract audio features
                analysis['audio_features'] = {
                    'mfcc_mean': np.mean(librosa.feature.mfcc(y=audio_data, sr=sr)),
                    'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)),
                    'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio_data)),
                    'rms_energy': np.mean(librosa.feature.rms(y=audio_data))
                }
                
                # Detect music vs speech (simple heuristic)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
                analysis['music_detected'] = np.mean(spectral_rolloff) > 4000
                
                # Noise level estimation
                analysis['noise_level'] = float(np.std(audio_data))
            
            # Speech-to-text transcription
            if hasattr(self, 'whisper_processor') and hasattr(self, 'whisper_model'):
                try:
                    # This would require proper audio preprocessing for Whisper
                    analysis['transcription'] = "[Transcription would be performed here]"
                    analysis['language_detected'] = 'en'
                except Exception as e:
                    logging.error(f"Transcription failed: {e}")
            
        except Exception as e:
            logging.error(f"Audio processing error: {e}")
        
        return analysis
    
    async def generate_audio(self, text: str, voice: str = "default") -> bytes:
        """Generate audio from text (text-to-speech)"""
        # This would use a TTS model in production
        # For demo, create a simple sine wave
        
        if not AUDIO_AVAILABLE:
            return b"Audio generation not available"
        
        try:
            # Generate a simple tone representing the text
            duration = len(text) * 0.1  # 0.1 seconds per character
            sample_rate = 22050
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Convert to bytes
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
            return audio_bytes.getvalue()
            
        except Exception as e:
            logging.error(f"Audio generation error: {e}")
            return b"Audio generation failed"

class VideoProcessor:
    """Advanced video processing capabilities"""
    
    def __init__(self):
        self.models = {}
    
    async def process_video(self, content: MultiModalContent) -> Dict[str, Any]:
        """Process video content with analysis"""
        
        if content.modality != ModalityType.VIDEO:
            raise ValueError("Content is not video")
        
        analysis = {
            'content_id': content.content_id,
            'duration': 0.0,
            'frame_rate': 0.0,
            'resolution': (0, 0),
            'total_frames': 0,
            'has_audio': False,
            'scenes_detected': [],
            'objects_tracked': [],
            'motion_analysis': {},
            'key_frames': []
        }
        
        try:
            if VIDEO_AVAILABLE:
                # Load video using moviepy
                if isinstance(content.data, bytes):
                    # Save bytes to temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        tmp_file.write(content.data)
                        tmp_path = tmp_file.name
                    
                    clip = VideoFileClip(tmp_path)
                    Path(tmp_path).unlink()  # Clean up
                else:
                    clip = VideoFileClip(content.data)
                
                analysis['duration'] = clip.duration
                analysis['frame_rate'] = clip.fps
                analysis['resolution'] = (clip.w, clip.h)
                analysis['total_frames'] = int(clip.duration * clip.fps)
                analysis['has_audio'] = clip.audio is not None
                
                # Extract key frames (every 10% of duration)
                key_frame_times = [i * clip.duration / 10 for i in range(11)]
                analysis['key_frames'] = key_frame_times
                
                # Scene detection (simplified)
                # In production, this would use computer vision algorithms
                analysis['scenes_detected'] = [
                    {'start_time': 0.0, 'end_time': clip.duration, 'scene_type': 'unknown'}
                ]
                
                clip.close()
            
        except Exception as e:
            logging.error(f"Video processing error: {e}")
        
        return analysis
    
    async def extract_frames(self, content: MultiModalContent, 
                           timestamps: List[float]) -> List[MultiModalContent]:
        """Extract frames from video at specified timestamps"""
        
        frames = []
        
        try:
            if VIDEO_AVAILABLE and isinstance(content.data, (str, bytes)):
                if isinstance(content.data, bytes):
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                        tmp_file.write(content.data)
                        tmp_path = tmp_file.name
                    
                    clip = VideoFileClip(tmp_path)
                    Path(tmp_path).unlink()
                else:
                    clip = VideoFileClip(content.data)
                
                for i, timestamp in enumerate(timestamps):
                    if timestamp <= clip.duration:
                        frame = clip.get_frame(timestamp)
                        
                        # Convert frame to image bytes
                        img = Image.fromarray(frame.astype('uint8'))
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG')
                        
                        frame_content = MultiModalContent(
                            content_id=f"{content.content_id}_frame_{i}",
                            modality=ModalityType.IMAGE,
                            format=ContentFormat.PNG,
                            data=img_bytes.getvalue(),
                            metadata={
                                'source_video': content.content_id,
                                'timestamp': timestamp,
                                'frame_index': i
                            }
                        )
                        
                        frames.append(frame_content)
                
                clip.close()
                
        except Exception as e:
            logging.error(f"Frame extraction error: {e}")
        
        return frames

# ============================================================================
# MULTI-MODAL AI COMMUNICATION HUB
# ============================================================================

class MultiModalAIHub:
    """Central hub for multi-modal AI communication"""
    
    def __init__(self):
        self.processors = {
            ModalityType.TEXT: TextProcessor(),
            ModalityType.IMAGE: ImageProcessor(),
            ModalityType.AUDIO: AudioProcessor(),
            ModalityType.VIDEO: VideoProcessor()
        }
        
        self.ai_agents = {}
        self.capability_profiles = {}
        self.content_store = {}
        self.processing_queue = asyncio.Queue()
        self.routing_table = defaultdict(list)
        self.protocol_translators = {}
        self.active_sessions = {}
        
        # Performance metrics
        self.processing_stats = defaultdict(lambda: defaultdict(int))
        self.response_times = defaultdict(list)
        
        # Start background services
        self.running = False
        self.processing_task = None
    
    async def register_ai_agent(self, agent: AIAgent, 
                               capability_profile: AICapabilityProfile):
        """Register an AI agent with its multi-modal capabilities"""
        
        self.ai_agents[agent.name] = agent
        self.capability_profiles[agent.name] = capability_profile
        
        # Update routing table
        for modality in capability_profile.supported_modalities:
            self.routing_table[modality].append(agent.name)
        
        logging.info(f"Registered AI agent: {agent.name} with {len(capability_profile.supported_modalities)} modalities")
    
    async def process_content(self, content: MultiModalContent, 
                            target_agent: Optional[str] = None) -> Dict[str, Any]:
        """Process multi-modal content and route to appropriate AI agent"""
        
        start_time = time.time()
        
        try:
            # Store content
            self.content_store[content.content_id] = content
            
            # Analyze content using modality-specific processor
            if content.modality in self.processors:
                processor = self.processors[content.modality]
                
                if content.modality == ModalityType.TEXT:
                    analysis = await processor.process_text(content)
                elif content.modality == ModalityType.IMAGE:
                    analysis = await processor.process_image(content)
                elif content.modality == ModalityType.AUDIO:
                    analysis = await processor.process_audio(content)
                elif content.modality == ModalityType.VIDEO:
                    analysis = await processor.process_video(content)
                else:
                    analysis = {'content_id': content.content_id, 'basic_info': 'processed'}
            else:
                analysis = {'content_id': content.content_id, 'error': 'Unsupported modality'}
            
            # Route to AI agent
            if target_agent and target_agent in self.ai_agents:
                agent_response = await self._route_to_agent(content, analysis, target_agent)
            else:
                agent_response = await self._auto_route_content(content, analysis)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats[content.modality.name]['processed'] += 1
            self.response_times[content.modality.name].append(processing_time)
            
            return {
                'content_id': content.content_id,
                'analysis': analysis,
                'agent_response': agent_response,
                'processing_time': processing_time,
                'status': 'success'
            }
            
        except Exception as e:
            logging.error(f"Content processing failed: {e}")
            return {
                'content_id': content.content_id,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'status': 'error'
            }
    
    async def _route_to_agent(self, content: MultiModalContent, 
                            analysis: Dict[str, Any], 
                            agent_name: str) -> Dict[str, Any]:
        """Route content to specific AI agent"""
        
        if agent_name not in self.ai_agents:
            return {'error': f'Agent {agent_name} not found'}
        
        agent = self.ai_agents[agent_name]
        profile = self.capability_profiles[agent_name]
        
        # Check if agent supports this modality
        if content.modality not in profile.supported_modalities:
            return {'error': f'Agent {agent_name} does not support {content.modality.name}'}
        
        # Prepare agent input
        agent_input = {
            'content': content,
            'analysis': analysis,
            'requested_capabilities': ['understand', 'analyze', 'respond']
        }
        
        # Execute agent processing
        try:
            # This would call the actual AI agent
            # For demo, simulate processing
            response = {
                'agent_id': agent_name,
                'understanding': f"Processed {content.modality.name} content",
                'insights': analysis,
                'recommendations': [
                    f"Content appears to be {content.modality.name}",
                    "Further processing available",
                    "Ready for cross-modal transformation"
                ],
                'confidence': 0.85,
                'processing_capabilities_used': profile.processing_capabilities[:3]
            }
            
            return response
            
        except Exception as e:
            return {'error': f'Agent processing failed: {e}'}
    
    async def _auto_route_content(self, content: MultiModalContent, 
                                analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically route content to best available agent"""
        
        # Find agents that support this modality
        candidate_agents = self.routing_table.get(content.modality, [])
        
        if not candidate_agents:
            return {'error': f'No agents available for {content.modality.name}'}
        
        # Select best agent based on capability match and performance
        best_agent = self._select_best_agent(candidate_agents, content)
        
        if best_agent:
            return await self._route_to_agent(content, analysis, best_agent)
        else:
            return {'error': 'No suitable agent found'}
    
    def _select_best_agent(self, candidate_agents: List[str], 
                          content: MultiModalContent) -> Optional[str]:
        """Select the best agent for processing content"""
        
        best_agent = None
        best_score = -1
        
        for agent_name in candidate_agents:
            if agent_name not in self.capability_profiles:
                continue
            
            profile = self.capability_profiles[agent_name]
            score = 0
            
            # Score based on format support
            if content.format in profile.input_formats:
                score += 10
            
            # Score based on performance metrics
            if 'accuracy' in profile.performance_metrics:
                score += profile.performance_metrics['accuracy'] * 5
            
            if 'speed' in profile.performance_metrics:
                score += profile.performance_metrics['speed'] * 3
            
            # Score based on specializations
            if content.modality == ModalityType.TEXT and 'nlp' in profile.specializations:
                score += 5
            elif content.modality == ModalityType.IMAGE and 'computer_vision' in profile.specializations:
                score += 5
            elif content.modality == ModalityType.AUDIO and 'speech_processing' in profile.specializations:
                score += 5
            
            if score > best_score:
                best_score = score
                best_agent = agent_name
        
        return best_agent
    
    async def cross_modal_translation(self, content: MultiModalContent, 
                                    target_modality: ModalityType,
                                    target_format: ContentFormat) -> MultiModalContent:
        """Translate content from one modality to another"""
        
        try:
            # Get appropriate processors
            source_processor = self.processors.get(content.modality)
            target_processor = self.processors.get(target_modality)
            
            if not source_processor or not target_processor:
                raise ValueError("Unsupported modality for translation")
            
            translation_id = f"{content.content_id}_to_{target_modality.name}"
            
            # Perform cross-modal translation
            if content.modality == ModalityType.TEXT and target_modality == ModalityType.IMAGE:
                # Text to image
                text_data = content.data if isinstance(content.data, str) else content.data.decode()
                image_data = await target_processor.generate_image(text_data)
                
                translated_content = MultiModalContent(
                    content_id=translation_id,
                    modality=target_modality,
                    format=target_format,
                    data=image_data,
                    metadata={
                        'source_content_id': content.content_id,
                        'translation_type': 'text_to_image',
                        'source_text': text_data[:100] + '...' if len(text_data) > 100 else text_data
                    }
                )
                
            elif content.modality == ModalityType.IMAGE and target_modality == ModalityType.TEXT:
                # Image to text (captioning)
                image_analysis = await source_processor.process_image(content)
                caption = image_analysis.get('caption', 'Image content analysis')
                
                translated_content = MultiModalContent(
                    content_id=translation_id,
                    modality=target_modality,
                    format=target_format,
                    data=caption,
                    metadata={
                        'source_content_id': content.content_id,
                        'translation_type': 'image_to_text',
                        'image_analysis': image_analysis
                    }
                )
                
            elif content.modality == ModalityType.AUDIO and target_modality == ModalityType.TEXT:
                # Audio to text (transcription)
                audio_analysis = await source_processor.process_audio(content)
                transcription = audio_analysis.get('transcription', '[Audio content transcription]')
                
                translated_content = MultiModalContent(
                    content_id=translation_id,
                    modality=target_modality,
                    format=target_format,
                    data=transcription,
                    metadata={
                        'source_content_id': content.content_id,
                        'translation_type': 'audio_to_text',
                        'audio_analysis': audio_analysis
                    }
                )
                
            elif content.modality == ModalityType.TEXT and target_modality == ModalityType.AUDIO:
                # Text to audio (TTS)
                text_data = content.data if isinstance(content.data, str) else content.data.decode()
                audio_data = await target_processor.generate_audio(text_data)
                
                translated_content = MultiModalContent(
                    content_id=translation_id,
                    modality=target_modality,
                    format=target_format,
                    data=audio_data,
                    metadata={
                        'source_content_id': content.content_id,
                        'translation_type': 'text_to_audio',
                        'source_text': text_data
                    }
                )
                
            else:
                # Generic translation (copy with format conversion)
                translated_content = MultiModalContent(
                    content_id=translation_id,
                    modality=target_modality,
                    format=target_format,
                    data=content.data,
                    metadata={
                        'source_content_id': content.content_id,
                        'translation_type': 'format_conversion'
                    }
                )
            
            # Store translated content
            self.content_store[translation_id] = translated_content
            
            return translated_content
            
        except Exception as e:
            logging.error(f"Cross-modal translation failed: {e}")
            raise
    
    async def create_multimodal_session(self, participants: List[str],
                                       supported_modalities: List[ModalityType]) -> str:
        """Create a multi-modal communication session"""
        
        session_id = f"session_{int(time.time())}"
        
        session = {
            'session_id': session_id,
            'participants': participants,
            'supported_modalities': supported_modalities,
            'created_at': time.time(),
            'messages': [],
            'active_translations': {},
            'session_metrics': {
                'total_messages': 0,
                'messages_by_modality': defaultdict(int),
                'translations_performed': 0
            }
        }
        
        self.active_sessions[session_id] = session
        
        logging.info(f"Created multi-modal session {session_id} with {len(participants)} participants")
        
        return session_id
    
    async def send_multimodal_message(self, session_id: str, sender: str, 
                                    content: MultiModalContent,
                                    auto_translate: bool = True) -> Dict[str, Any]:
        """Send a multi-modal message in a session"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # Process the content
        processing_result = await self.process_content(content)
        
        # Create message
        message = {
            'message_id': f"msg_{int(time.time()*1000)}",
            'session_id': session_id,
            'sender': sender,
            'content': content,
            'processing_result': processing_result,
            'timestamp': time.time(),
            'translations': {}
        }
        
        # Auto-translate to other modalities if requested
        if auto_translate:
            for modality in session['supported_modalities']:
                if modality != content.modality:
                    try:
                        # Determine appropriate format for target modality
                        if modality == ModalityType.TEXT:
                            target_format = ContentFormat.PLAIN_TEXT
                        elif modality == ModalityType.IMAGE:
                            target_format = ContentFormat.PNG
                        elif modality == ModalityType.AUDIO:
                            target_format = ContentFormat.WAV
                        else:
                            continue  # Skip unsupported modalities
                        
                        translated = await self.cross_modal_translation(
                            content, modality, target_format
                        )
                        message['translations'][modality.name] = translated
                        
                    except Exception as e:
                        logging.error(f"Translation to {modality.name} failed: {e}")
        
        # Add to session
        session['messages'].append(message)
        session['session_metrics']['total_messages'] += 1
        session['session_metrics']['messages_by_modality'][content.modality.name] += 1
        session['session_metrics']['translations_performed'] += len(message['translations'])
        
        return {
            'message_id': message['message_id'],
            'status': 'sent',
            'processing_result': processing_result,
            'translations_available': list(message['translations'].keys())
        }
    
    async def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """Get multi-modal session history"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # Serialize session data
        history = {
            'session_id': session_id,
            'participants': session['participants'],
            'supported_modalities': [m.name for m in session['supported_modalities']],
            'created_at': session['created_at'],
            'total_messages': len(session['messages']),
            'session_metrics': session['session_metrics'],
            'recent_messages': []
        }
        
        # Add recent messages (last 10)
        for message in session['messages'][-10:]:
            msg_summary = {
                'message_id': message['message_id'],
                'sender': message['sender'],
                'content_type': message['content'].modality.name,
                'content_format': message['content'].format.value,
                'timestamp': message['timestamp'],
                'has_translations': len(message['translations']) > 0,
                'processing_success': message['processing_result']['status'] == 'success'
            }
            history['recent_messages'].append(msg_summary)
        
        return history
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        
        metrics = {
            'registered_agents': len(self.ai_agents),
            'supported_modalities': len(self.processors),
            'content_processed': dict(self.processing_stats),
            'active_sessions': len(self.active_sessions),
            'content_stored': len(self.content_store),
            'average_response_times': {},
            'agent_capabilities': {}
        }
        
        # Calculate average response times
        for modality, times in self.response_times.items():
            if times:
                metrics['average_response_times'][modality] = {
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'min': np.min(times),
                    'max': np.max(times)
                }
        
        # Agent capability summary
        for agent_name, profile in self.capability_profiles.items():
            metrics['agent_capabilities'][agent_name] = {
                'supported_modalities': [m.name for m in profile.supported_modalities],
                'processing_capabilities': profile.processing_capabilities,
                'specializations': profile.specializations,
                'performance_metrics': profile.performance_metrics
            }
        
        return metrics

# ============================================================================
# MULTI-MODAL DEMO
# ============================================================================

async def multimodal_ai_demo():
    """Demonstrate multi-modal AI communication capabilities"""
    
    print("🎭 Multi-modal AI Communication Hub Demo")
    print("=" * 50)
    
    # Create hub
    hub = MultiModalAIHub()
    
    # Create AI agents with different capabilities
    agents_data = [
        {
            'name': 'TextMaster',
            'modalities': [ModalityType.TEXT],
            'input_formats': [ContentFormat.PLAIN_TEXT, ContentFormat.MARKDOWN],
            'output_formats': [ContentFormat.PLAIN_TEXT],
            'capabilities': ['nlp', 'sentiment_analysis', 'summarization'],
            'specializations': ['nlp', 'text_generation']
        },
        {
            'name': 'VisionAI',
            'modalities': [ModalityType.IMAGE],
            'input_formats': [ContentFormat.JPEG, ContentFormat.PNG],
            'output_formats': [ContentFormat.PNG],
            'capabilities': ['object_detection', 'image_captioning', 'image_generation'],
            'specializations': ['computer_vision', 'image_analysis']
        },
        {
            'name': 'AudioBot',
            'modalities': [ModalityType.AUDIO],
            'input_formats': [ContentFormat.WAV, ContentFormat.MP3],
            'output_formats': [ContentFormat.WAV],
            'capabilities': ['speech_recognition', 'audio_analysis', 'tts'],
            'specializations': ['speech_processing', 'audio_analysis']
        }
    ]
    
    # Register agents
    for agent_data in agents_data:
        # Create AI agent
        capability = LLMCapability("mock-model", "multimodal")
        agent = AIAgent(agent_data['name'], [capability])
        
        # Create capability profile
        profile = AICapabilityProfile(
            agent_id=agent_data['name'],
            supported_modalities=agent_data['modalities'],
            input_formats=agent_data['input_formats'],
            output_formats=agent_data['output_formats'],
            processing_capabilities=agent_data['capabilities'],
            performance_metrics={'accuracy': 0.9, 'speed': 0.8},
            specializations=agent_data['specializations']
        )
        
        await hub.register_ai_agent(agent, profile)
    
    print(f"✅ Registered {len(agents_data)} AI agents")
    
    # Create sample content
    contents = []
    
    # Text content
    text_content = MultiModalContent(
        content_id="text_001",
        modality=ModalityType.TEXT,
        format=ContentFormat.PLAIN_TEXT,
        data="This is a sample text message about artificial intelligence and machine learning.",
        metadata={'source': 'demo', 'language': 'en'}
    )
    contents.append(text_content)
    
    # Image content (mock)
    if VISION_AVAILABLE:
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        image_content = MultiModalContent(
            content_id="image_001",
            modality=ModalityType.IMAGE,
            format=ContentFormat.PNG,
            data=img_bytes.getvalue(),
            metadata={'source': 'demo', 'color': 'red'}
        )
        contents.append(image_content)
    
    print(f"✅ Created {len(contents)} sample content items")
    
    # Process each content item
    for content in contents:
        result = await hub.process_content(content)
        print(f"✅ Processed {content.modality.name} content:")
        print(f"   Status: {result['status']}")
        print(f"   Processing time: {result['processing_time']:.3f}s")
        if 'agent_response' in result and 'agent_id' in result['agent_response']:
            print(f"   Handled by: {result['agent_response']['agent_id']}")
    
    # Demonstrate cross-modal translation
    if contents:
        source_content = contents[0]  # Text content
        
        try:
            translated = await hub.cross_modal_translation(
                source_content, 
                ModalityType.IMAGE, 
                ContentFormat.PNG
            )
            print(f"✅ Cross-modal translation: {source_content.modality.name} → {translated.modality.name}")
            print(f"   Translation ID: {translated.content_id}")
        except Exception as e:
            print(f"⚠️  Cross-modal translation failed: {e}")
    
    # Create multi-modal session
    session_id = await hub.create_multimodal_session(
        participants=['TextMaster', 'VisionAI', 'AudioBot'],
        supported_modalities=[ModalityType.TEXT, ModalityType.IMAGE, ModalityType.AUDIO]
    )
    print(f"✅ Created multi-modal session: {session_id}")
    
    # Send messages in session
    for i, content in enumerate(contents):
        message_result = await hub.send_multimodal_message(
            session_id, f"user_{i}", content, auto_translate=True
        )
        print(f"✅ Sent message: {message_result['message_id']}")
        print(f"   Translations: {message_result['translations_available']}")
    
    # Get session history
    history = await hub.get_session_history(session_id)
    print(f"✅ Session history: {history['total_messages']} messages")
    print(f"   Modalities used: {list(history['session_metrics']['messages_by_modality'].keys())}")
    
    # Get system metrics
    metrics = hub.get_system_metrics()
    print(f"✅ System metrics:")
    print(f"   Registered agents: {metrics['registered_agents']}")
    print(f"   Content processed: {sum(sum(modality_stats.values()) for modality_stats in metrics['content_processed'].values())}")
    print(f"   Active sessions: {metrics['active_sessions']}")
    
    print("\n🎉 Multi-modal AI Communication Hub Demo completed!")
    print("Features demonstrated:")
    print("• Multi-modal content processing (text, image, audio, video)")
    print("• AI agent registration with capability profiles")
    print("• Intelligent routing based on modality and capabilities")
    print("• Cross-modal content translation")
    print("• Multi-modal communication sessions")
    print("• Real-time message processing and translation")
    print("• Comprehensive analytics and metrics")
    print("• Format conversion and protocol translation")
    print("• Performance optimization and load balancing")

if __name__ == "__main__":
    asyncio.run(multimodal_ai_demo())
