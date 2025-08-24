"""
A2A-Compliant Perplexity AI API Module for Agent 0
Implements blockchain-based messaging for external API integration

A2A PROTOCOL COMPLIANCE:
This module has been modified to comply with A2A protocol requirements.
All external API calls must be routed through the A2A blockchain messaging system
to maintain protocol compliance and audit trails.
"""

import asyncio
# REMOVED: import aiohttp  # A2A Protocol Violation - direct HTTP not allowed
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
# REMOVED: from aiohttp_retry import RetryClient, ExponentialRetry  # A2A Protocol Violation
import backoff

# NLP libraries for sentiment analysis
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sentiment analysis libraries not available: {e}")
    SENTIMENT_AVAILABLE = False

# Grok AI client for enhanced analysis
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reasoningAgent'))
    from asyncGrokClient import AsyncGrokReasoning, GrokConfig
    GROK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Grok client not available: {e}")
    GROK_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerplexityAPIClient:
    """A2A-Compliant Perplexity AI API client using blockchain messaging"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.perplexity.ai"):
        self.api_key = api_key
        self.base_url = base_url
        # REMOVED: self.session = None  # A2A Protocol Violation - direct HTTP not allowed
        
        # A2A Compliance: Initialize blockchain messaging client for external API gateway
        self.a2a_client = None
        self._initialize_a2a_client()
    
    def _initialize_a2a_client(self):
        """Initialize A2A blockchain messaging client for protocol compliance"""
        try:
            # Import A2A SDK components
            from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
            from ....sdk.a2aNetworkClient import A2ANetworkClient
            
            # Initialize A2A client for blockchain-based external API routing
            self.a2a_client = A2ANetworkClient(
                agent_id="agent0_perplexity_gateway",
                private_key=os.getenv('A2A_PRIVATE_KEY'),
                rpc_url=os.getenv('A2A_RPC_URL', 'http://localhost:8545')
            )
            logger.info("A2A blockchain client initialized for Perplexity API gateway")
        except Exception as e:
            logger.error(f"Failed to initialize A2A client: {e}")
            # In compliance mode, we cannot proceed without A2A client
            raise RuntimeError("A2A Protocol Violation: Cannot operate without blockchain messaging client")
        
        # Initialize Grok client for enhanced analysis
        self.grok_client = None
        if GROK_AVAILABLE:
            try:
                grok_config = GrokConfig(
                    api_key=os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'),
                    cache_ttl=300,  # 5 minutes cache for news analysis
                    pool_connections=5
                )
                self.grok_client = AsyncGrokReasoning(grok_config)
                logger.info("Grok client initialized for news analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok client: {e}")
                self.grok_client = None
        self.rate_limiter = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        # Initialize sentiment analyzers
        if SENTIMENT_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Source credibility database
        self.credibility_scores = {
            "reuters.com": 0.95,
            "ap.org": 0.94,
            "bbc.com": 0.92,
            "npr.org": 0.91,
            "wsj.com": 0.90,
            "nytimes.com": 0.89,
            "washingtonpost.com": 0.88,
            "cnn.com": 0.85,
            "bloomberg.com": 0.87,
            "ft.com": 0.89,
            "economist.com": 0.88,
            "perplexity.ai": 0.85,
            "default": 0.70
        }
    
    async def __aenter__(self):
        """A2A-Compliant async context manager entry"""
        # A2A Protocol: No direct HTTP session initialization
        # All communication goes through blockchain messaging
        logger.info("A2A Protocol: Blockchain messaging client ready for external API requests")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """A2A-Compliant async context manager exit"""
        # A2A Protocol: Clean up A2A client connections if needed
        if self.a2a_client:
            try:
                await self.a2a_client.close()
            except Exception as e:
                logger.warning(f"A2A client cleanup warning: {e}")
        logger.info("A2A Protocol: Blockchain messaging client connections closed")
    
    @backoff.on_exception(
        backoff.expo,
        (RuntimeError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def search_news(self, query: str, filters: List[str] = None, 
                         date_range: str = "today", max_articles: int = 10) -> Dict[str, Any]:
        """
        A2A-Compliant news search using blockchain messaging
        Routes external API requests through A2A protocol for audit and compliance
        """
        async with self.rate_limiter:
            try:
                # Construct search prompt
                search_prompt = self._build_search_prompt(query, filters, date_range, max_articles)
                
                # A2A Protocol: Create blockchain message for external API request
                api_request_payload = {
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a news aggregator. Return structured news data in JSON format with title, content, source, url, and timestamp for each article."
                        },
                        {
                            "role": "user",
                            "content": search_prompt
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "return_citations": True,
                    "search_domain_filter": ["perplexity.ai"],
                    "return_images": False
                }
                
                # A2A Protocol: Route API request through blockchain messaging
                try:
                    # Create A2A message for external API gateway
                    a2a_message = {
                        "message_type": "external_api_request",
                        "target_service": "perplexity_api",
                        "endpoint": f"{self.base_url}/chat/completions",
                        "method": "POST",
                        "headers": {
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        "payload": api_request_payload,
                        "requester_agent": "agent0_perplexity_gateway",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send message through A2A blockchain network to external API gateway
                    response = await self.a2a_client.send_external_api_request(a2a_message)
                    
                    if response.get("status") == "success":
                        return await self._process_perplexity_response(response.get("data"), query)
                    else:
                        error_msg = response.get("error", "Unknown API error")
                        logger.error(f"A2A External API gateway error: {error_msg}")
                        return {"articles": [], "error": f"A2A API error: {error_msg}"}
                        
                except Exception as a2a_error:
                    logger.error(f"A2A blockchain messaging failed: {a2a_error}")
                    # A2A Protocol: No HTTP fallback allowed - must use blockchain messaging
                    return {
                        "articles": [], 
                        "error": f"A2A Protocol Compliance: External API request must route through blockchain. Error: {a2a_error}"
                    }
                        
            except Exception as e:
                logger.error(f"A2A-compliant API request failed: {e}")
                return {"articles": [], "error": str(e)}
    
    def _build_search_prompt(self, query: str, filters: List[str], date_range: str, max_articles: int) -> str:
        """Build search prompt for Perplexity AI"""
        prompt_parts = [
            f"Find {max_articles} recent news articles about: {query}"
        ]
        
        if filters:
            prompt_parts.append(f"Focus on these topics: {', '.join(filters)}")
        
        date_filter = {
            "today": "from today",
            "week": "from the past week", 
            "month": "from the past month"
        }.get(date_range, "recent")
        
        prompt_parts.append(f"Only include articles {date_filter}")
        prompt_parts.append("Return as JSON array with fields: title, content, source, url, timestamp, relevance_score")
        
        return ". ".join(prompt_parts)
    
    async def _process_perplexity_response(self, response: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Process Perplexity AI response and extract articles"""
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response.get("citations", [])
            
            # Try to extract JSON from response
            articles = []
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            if json_match:
                try:
                    articles_data = json.loads(json_match.group())
                    for article_data in articles_data:
                        if isinstance(article_data, dict):
                            articles.append(await self._enrich_article(article_data, citations))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from Perplexity response")
            
            # Fallback: extract from citations if JSON parsing fails
            if not articles and citations:
                for citation in citations[:10]:  # Limit to 10 citations
                    article = await self._create_article_from_citation(citation, query)
                    if article:
                        articles.append(article)
            
            return {
                "articles": articles,
                "total_found": len(articles),
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process Perplexity response: {e}")
            return {"articles": [], "error": str(e)}
    
    async def _enrich_article(self, article_data: Dict[str, Any], citations: List[Dict]) -> Dict[str, Any]:
        """Enrich article with additional metadata"""
        # Extract domain from URL for credibility scoring
        url = article_data.get("url", "")
        domain = self._extract_domain(url)
        
        enriched_article = {
            "title": article_data.get("title", ""),
            "content": article_data.get("content", ""),
            "source": article_data.get("source", domain),
            "url": url,
            "timestamp": article_data.get("timestamp", datetime.utcnow().isoformat()),
            "relevance_score": article_data.get("relevance_score", 0.8),
            "domain": domain,
            "credibility_score": self.get_source_credibility(domain),
            "word_count": len(article_data.get("content", "").split()),
            "language": await self._detect_language(article_data.get("content", ""))
        }
        
        # Add sentiment analysis
        if SENTIMENT_AVAILABLE and enriched_article["content"]:
            enriched_article["sentiment"] = await self.analyze_sentiment(enriched_article["content"])
        
        return enriched_article
    
    async def _detect_language(self, text: str) -> str:
        """Detect language of text content"""
        if not text or len(text) < 10:
            return "en"  # Default to English for short texts
        
        try:
            # Use Grok AI for enhanced language detection
            if self.grok_client:
                prompt = f"""
                Detect the language of this text and return only the ISO 639-1 language code:
                
                Text: "{text[:200]}..."
                
                Return only the 2-letter language code (e.g., 'en', 'es', 'fr', 'de', 'zh', etc.)
                """
                
                result = await self.grok_client.decompose_question(prompt)
                if result.get("success"):
                    decomposition = result.get("decomposition", {})
                    lang_code = decomposition.get("language_code", "en")
                    if isinstance(lang_code, str) and len(lang_code) == 2:
                        return lang_code.lower()
            
            # Fallback: simple heuristic detection
            # Check for common non-English characters
            if any(ord(char) > 127 for char in text[:100]):
                # Contains non-ASCII characters, likely non-English
                chinese_chars = sum(1 for char in text[:100] if '\u4e00' <= char <= '\u9fff')
                if chinese_chars > 5:
                    return "zh"
                arabic_chars = sum(1 for char in text[:100] if '\u0600' <= char <= '\u06ff')
                if arabic_chars > 5:
                    return "ar"
                # Add more language detection logic as needed
                return "unknown"
            
            return "en"  # Default to English
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"
    
    async def _create_article_from_citation(self, citation: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Create article from Perplexity citation"""
        try:
            url = citation.get("url", "")
            title = citation.get("title", "")
            
            if not url or not title:
                return None
            
            # Try to extract more content using newspaper3k
            content = ""
            if NLP_AVAILABLE:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    content = article.text[:1000]  # Limit content length
                except:
                    content = citation.get("text", "")
            else:
                content = citation.get("text", "")
            
            domain = self._extract_domain(url)
            
            return {
                "title": title,
                "content": content,
                "source": domain,
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "relevance_score": 0.8,  # Default relevance
                "domain": domain,
                "credibility_score": self.get_source_credibility(domain),
                "word_count": len(content.split()),
                "language": "en",
                "sentiment": await self.analyze_sentiment(content) if NLP_AVAILABLE and content else None
            }
            
        except Exception as e:
            logger.warning(f"Failed to create article from citation: {e}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "unknown"
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple NLP approaches enhanced with Grok AI"""
        try:
            # Enhanced sentiment analysis with Grok AI
            if self.grok_client:
                grok_analysis = await self._analyze_sentiment_with_grok(text)
                if grok_analysis.get("success"):
                    return grok_analysis["result"]
            
            # Fallback to traditional NLP if Grok unavailable
            if not SENTIMENT_AVAILABLE:
                return {"sentiment": "neutral", "confidence": 0.5, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}}
            
            # VADER sentiment analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine results
            combined_sentiment = "neutral"
            if vader_scores['compound'] > 0.1 and textblob_polarity > 0.1:
                combined_sentiment = "positive"
            elif vader_scores['compound'] < -0.1 and textblob_polarity < -0.1:
                combined_sentiment = "negative"
            
            confidence = min(abs(vader_scores['compound']) + abs(textblob_polarity), 1.0) / 2
            
            return {
                "sentiment": combined_sentiment,
                "confidence": confidence,
                "scores": {
                    "positive": vader_scores['pos'],
                    "negative": vader_scores['neg'],
                    "neutral": vader_scores['neu']
                },
                "vader_compound": vader_scores['compound'],
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity,
                "method": "traditional_nlp"
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}}
    
    async def _analyze_sentiment_with_grok(self, text: str) -> Dict[str, Any]:
        """Enhanced sentiment analysis using Grok AI"""
        try:
            prompt = f"""
            Analyze the sentiment of this text with high precision:
            
            Text: "{text}"
            
            Provide detailed sentiment analysis including:
            1. Primary sentiment (positive/negative/neutral)
            2. Confidence score (0-1)
            3. Emotional indicators and reasoning
            4. Nuanced sentiment breakdown
            5. Context-aware interpretation
            
            Return as JSON with fields: sentiment, confidence, scores, reasoning, emotions
            """
            
            result = await self.grok_client.decompose_question(prompt)
            
            if result.get("success"):
                decomposition = result.get("decomposition", {})
                return {
                    "success": True,
                    "result": {
                        "sentiment": decomposition.get("sentiment", "neutral"),
                        "confidence": decomposition.get("confidence", 0.7),
                        "scores": decomposition.get("scores", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}),
                        "reasoning": decomposition.get("reasoning", ""),
                        "emotions": decomposition.get("emotions", []),
                        "method": "grok_ai",
                        "grok_cached": result.get("cached", False),
                        "response_time": result.get("response_time", 0)
                    }
                }
            else:
                return {"success": False}
                
        except Exception as e:
            logger.error(f"Grok sentiment analysis failed: {e}")
            return {"success": False}

    async def assess_source_credibility(self, source: str) -> float:
        """Assess news source credibility with enhanced Grok AI scoring"""
        try:
            # Enhanced credibility assessment with Grok AI
            if self.grok_client:
                grok_assessment = await self._assess_credibility_with_grok(source)
                if grok_assessment.get("success"):
                    return grok_assessment["credibility_score"]
            
            # Fallback to traditional domain-based scoring
            domain = self._extract_domain(source)
            base_score = self.get_source_credibility(domain)
            
            return min(max(base_score, 0.0), 1.0)  # Ensure 0-1 range
            
        except Exception as e:
            logger.error(f"Source credibility assessment failed: {e}")
            return 0.5  # Default neutral score
    
    async def _assess_credibility_with_grok(self, source: str) -> Dict[str, Any]:
        """Enhanced credibility assessment using Grok AI"""
        try:
            prompt = f"""
            Assess the credibility of this news source with comprehensive analysis:
            
            Source: "{source}"
            
            Evaluate based on:
            1. Domain reputation and history
            2. Editorial standards and fact-checking
            3. Bias indicators and transparency
            4. Professional journalism standards
            5. Recent reliability metrics
            
            Provide credibility score (0-1) and detailed reasoning.
            Return as JSON with fields: credibility_score, reasoning, factors, recommendations
            """
            
            result = await self.grok_client.analyze_patterns(prompt)
            
            if result.get("success"):
                patterns = result.get("patterns", {})
                return {
                    "success": True,
                    "credibility_score": patterns.get("credibility_score", 0.7),
                    "reasoning": patterns.get("reasoning", ""),
                    "factors": patterns.get("factors", []),
                    "grok_cached": result.get("cached", False),
                    "response_time": result.get("response_time", 0)
                }
            else:
                return {"success": False}
                
        except Exception as e:
            logger.error(f"Grok credibility assessment failed: {e}")
            return {"success": False}

    def get_source_credibility(self, domain: str) -> float:
        """Get credibility score for news source"""
        domain = domain.lower()
        
        # Check exact match first
        if domain in self.credibility_scores:
            return self.credibility_scores[domain]
        
        # Check for partial matches (e.g., subdomain)
        for known_domain, score in self.credibility_scores.items():
            if known_domain != "default" and known_domain in domain:
                return score
        
        # Return default score
        return self.credibility_scores["default"]
    
    async def update_credibility_database(self, domain: str, score: float, source: str = "manual"):
        """Update credibility score for a domain"""
        if 0.0 <= score <= 1.0:
            self.credibility_scores[domain.lower()] = score
            logger.info(f"Updated credibility score for {domain}: {score} (source: {source})")
        else:
            logger.warning(f"Invalid credibility score {score} for {domain}")
    
    async def get_trending_topics(self, category: str = "general") -> List[str]:
        """Get trending topics using Grok AI analysis"""
        if not self.grok_client:
            return ["artificial intelligence", "machine learning", "blockchain technology", "climate change", "renewable energy"]
        
        try:
            prompt = f"""
            Identify current trending topics in the {category} category based on recent news patterns and social media discussions.
            
            Provide 5-10 trending topics that are:
            1. Currently relevant and newsworthy
            2. Generating significant discussion
            3. Appropriate for news collection
            4. Diverse across different domains
            
            Return as JSON array of topic strings.
            """
            
            result = await self.grok_client.analyze_patterns(prompt)
            
            if result.get("success"):
                patterns = result.get("patterns", {})
                topics = patterns.get("trending_topics", [])
                if topics and isinstance(topics, list):
                    return topics[:10]  # Limit to 10 topics
            
            # Fallback to default topics
            return ["artificial intelligence", "machine learning", "blockchain technology", "climate change", "renewable energy"]
            
        except Exception as e:
            logger.error(f"Trending topics analysis failed: {e}")
            return ["artificial intelligence", "machine learning", "blockchain technology", "climate change", "renewable energy"]

class EnhancedNewsProcessor:
    """Enhanced news processing with advanced features"""
    
    def __init__(self):
        self.article_cache = {}
        self.duplicate_threshold = 0.8
    
    async def detect_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and remove duplicate articles"""
        if not SENTIMENT_AVAILABLE:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get("title", "").lower()
            
            # Enhanced duplicate detection by title similarity
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = self._calculate_similarity(title, seen_title)
                if similarity > self.duplicate_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.add(title)
        
        return unique_articles
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using enhanced word overlap and semantic analysis"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Enhanced with character-level similarity for better matching
        char_similarity = self._character_similarity(text1, text2)
        
        # Weighted combination
        return (jaccard * 0.7) + (char_similarity * 0.3)
    
    def _character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap ratio
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        intersection = chars1.intersection(chars2)
        union = chars1.union(chars2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def categorize_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize articles by topic"""
        categories = {
            "technology": [],
            "business": [],
            "politics": [],
            "science": [],
            "health": [],
            "sports": [],
            "entertainment": [],
            "other": []
        }
        
        # Enhanced keyword-based categorization with Grok AI support
        category_keywords = {
            "technology": ["ai", "artificial intelligence", "tech", "software", "computer", "digital", "cyber"],
            "business": ["market", "stock", "economy", "financial", "business", "company", "corporate"],
            "politics": ["government", "election", "political", "policy", "congress", "senate", "president"],
            "science": ["research", "study", "scientific", "discovery", "experiment", "climate"],
            "health": ["health", "medical", "disease", "treatment", "vaccine", "hospital", "doctor"],
            "sports": ["game", "team", "player", "championship", "league", "sport", "match"],
            "entertainment": ["movie", "music", "celebrity", "film", "show", "entertainment", "actor"]
        }
        
        for article in articles:
            content = (article.get("title", "") + " " + article.get("content", "")).lower()
            
            best_category = "other"
            max_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in content)
                if matches > max_matches:
                    max_matches = matches
                    best_category = category
            
            categories[best_category].append(article)
        
        return categories
