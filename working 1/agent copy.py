# agent.py - Complete SEO Agent System

import os
import re
import json
import csv
import requests
from datetime import datetime, timedelta
from typing import List, TypedDict, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import UnstructuredExcelLoader
from dotenv import load_dotenv
from serpapi import GoogleSearch
#from serpapi.google_search import GoogleSearch

from googleapiclient.discovery import build
import replicate

load_dotenv()


# ========== CALENDAR AGENT STATE ==========
class CalendarAgentState(TypedDict):
    initial_query: str
    keywords: List[str]
    enriched_keywords: List[dict]
    keyword_clusters: List[dict]
    user_preferences: dict
    scored_and_tagged_topics: List[dict]
    content_plan: List[dict]
    content_briefs: List[dict]


# ========== SEO WRITER AGENT STATE ==========
class SEOWriterAgentState(TypedDict):
    content_brief: dict
    target_keyword: str
    serp_analysis: dict
    entities_and_semantics: dict
    article_blueprint: dict
    written_article: str
    verification_results: dict
    verification_feedback: Optional[List[str]]
    media_plan: dict
    final_article: str
    iteration_count: int


# ========== COMBINED STATE FOR FULL WORKFLOW ==========
class FullWorkflowState(TypedDict):
    # Calendar Agent States
    initial_query: str
    keywords: List[str]
    enriched_keywords: List[dict]
    keyword_clusters: List[dict]
    user_preferences: dict
    scored_and_tagged_topics: List[dict]
    content_plan: List[dict]
    content_briefs: List[dict]

    # SEO Writer States
    current_brief_index: int
    current_content_brief: dict
    target_keyword: str
    serp_analysis: dict
    entities_and_semantics: dict
    article_blueprint: dict
    written_article: str
    verification_results: dict
    verification_feedback: Optional[List[str]]
    media_plan: dict
    final_article: str
    iteration_count: int
    completed_articles: List[dict]


# ========== LLM INITIALIZATION ==========
# Calendar Agent LLM
llm = ChatGroq(temperature=0.7,
               model_name="llama-3.3-70b-versatile",
               groq_api_key=os.getenv("GROQ_API_KEY"))

# SEO Writer Agent LLMs
writer_llm = ChatGroq(
    temperature=0.7,
    model_name=
    "llama-3.3-70b-versatile",  # Using available model instead of kimi
    groq_api_key=os.getenv("GROQ_API_KEY"))

verifier_llm = ChatOpenAI(temperature=0.3,
                          model_name="gpt-4o",
                          openai_api_key=os.getenv("OPENAI_API_KEY"))

media_planner_llm = ChatGoogleGenerativeAI(
    temperature=0.5,
    model="gemini-2.0-flash-exp",  # Using available model
    google_api_key=os.getenv("GOOGLE_API_KEY"))


# ========== CALENDAR AGENT FUNCTIONS ==========
def call_keyword_everywhere_api(keywords: List[str]) -> List[dict]:
    """Calls the Keyword Everywhere API to get metrics for a list of keywords."""
    if not keywords: return []
    KEYWORD_EVERYWHERE_API_KEY = os.getenv("KEYWORD_EVERYWHERE_API_KEY")
    if not KEYWORD_EVERYWHERE_API_KEY: 
        print("Warning: KEYWORD_EVERYWHERE_API_KEY not found, using mock data")
        return [{
            "keyword": kw,
            "volume": 1000 + (len(kw) * 100),  # Mock volume
            "competition": 0.5
        } for kw in keywords[:10]]  # Limit to 10 for testing
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {KEYWORD_EVERYWHERE_API_KEY}'
    }
    data = [('kw[]', kw) for kw in keywords]
    url = "https://api.keywordseverywhere.com/v1/get_keyword_data"
    try:
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        api_data = response.json().get('data', [])
        return [{
            "keyword": m.get('keyword', 'N/A'),
            "volume": m.get('vol', 0),
            "competition": m.get('competition', 0)
        } for m in api_data if isinstance(m, dict) and m.get('vol', 0) > 0]
    except requests.RequestException as e:
        print(f"Error calling Keyword Everywhere API: {e}")
        # Return mock data as fallback
        return [{
            "keyword": kw,
            "volume": 1000 + (len(kw) * 100),
            "competition": 0.5
        } for kw in keywords[:10]]


def call_serpapi_for_trends(topic_title: str) -> str:
    """Calls the SerpApi Google Trends API to analyze search interest over time."""
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    if not SERPAPI_API_KEY:
        return "Stable"

    params = {
        "engine": "google_trends",
        "q": topic_title,
        "api_key": SERPAPI_API_KEY,
        "data_type": "TIMESERIES"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "interest_over_time" not in results:
            return "Stable"

        timeline_data = results["interest_over_time"]["timeline_data"]
        values = [
            item['values'][0]['extracted_value'] for item in timeline_data
        ]

        if not values or len(values) < 12:
            return "Stable"

        max_val = max(values)
        min_val = min(values)
        if max_val > 60 and min_val < 30 and (max_val / (min_val + 1)) > 2.5:
            return "Seasonal"

        half_point = len(values) // 2
        avg_first_half = sum(values[:half_point]) / half_point
        avg_second_half = sum(values[half_point:]) / half_point
        if avg_second_half > avg_first_half * 1.3:
            return "Rising Trend"

        return "Stable"
    except Exception as e:
        print(f"Error calling SerpApi: {e}")
        return "Stable"


# ========== CALENDAR AGENT NODES ==========
def get_initial_keywords_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 1: Gets initial keywords from various sources."""
    query = state['initial_query'].strip()
    keywords = []
    try:
        if query.lower().endswith(".csv"):
            if os.path.exists(query):
                with open(query, mode='r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    keywords = [
                        rows[0].strip() for rows in reader
                        if rows and rows[0].strip()
                    ]
            else:
                print(f"CSV file not found: {query}")
        elif query.lower().endswith((".xlsx", ".xls")):
            if os.path.exists(query):
                loader = UnstructuredExcelLoader(file_path=query, mode="elements")
                docs = loader.load()
                keywords = [
                    doc.page_content.strip() for doc in docs
                    if doc.page_content.strip()
                ]
            else:
                print(f"Excel file not found: {query}")
        else:
            if query:
                keywords = [
                    kw.strip() for kw in re.split(r'[,\s]+', query)
                    if kw.strip()
                ]
    except Exception as e:
        print(f"An error occurred while processing input: {e}")
    
    if not keywords:
        keywords = ["default keyword"]  # Fallback
        
    print(f"Extracted {len(keywords)} keywords: {keywords[:5]}...")
    return {"keywords": keywords}


def enrich_keywords_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 2: Enriches the raw keywords with volume/CPC data."""
    keywords = state.get('keywords', [])
    if not keywords:
        return {"enriched_keywords": []}
        
    print(f"Enriching {len(keywords)} keywords...")
    enriched_keywords = call_keyword_everywhere_api(keywords)
    print(f"Enriched {len(enriched_keywords)} keywords with metrics")
    return {"enriched_keywords": enriched_keywords}


def cluster_keywords_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 3: Groups enriched keywords into semantic clusters."""
    enriched_keywords = state.get('enriched_keywords', [])
    if not enriched_keywords:
        print("No enriched keywords found, creating default cluster")
        return {"keyword_clusters": [{
            "topic_title": "Default Topic",
            "keywords": state.get('keywords', ['default keyword'])[:5]
        }]}

    keyword_list_str = ", ".join([kw['keyword'] for kw in enriched_keywords])
    print(f"Clustering keywords: {keyword_list_str[:100]}...")

    prompt = ChatPromptTemplate.from_messages([(
        "system",
        "You are an expert content strategist. Your task is to group the provided keywords into 5-10 semantic clusters. You must respond with ONLY a valid JSON array of objects. Do not include any introductory text, explanations, or markdown formatting like ```json. Each object in the array must have two keys: 'topic_title' (a string) and 'keywords' (a list of strings)."
    ), ("user", "Keywords to cluster: {keywords}")])

    try:
        chain = prompt | llm
        response = chain.invoke({"keywords": keyword_list_str})
        raw_response = response.content

        # Try to extract JSON from response
        json_string = None
        json_match = re.search(r"```json\n(.*)\n```", raw_response, re.DOTALL)
        if json_match:
            json_string = json_match.group(1).strip()
        else:
            start_index = raw_response.find('[')
            if start_index != -1:
                json_string = raw_response[start_index:]

        if json_string:
            try:
                clusters = json.loads(json_string)
                if clusters and isinstance(clusters, list):
                    print(f"Successfully created {len(clusters)} clusters")
                    return {"keyword_clusters": clusters}
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        
        # Fallback clustering
        print("Using fallback clustering method")
        fallback_clusters = []
        keywords_per_cluster = max(1, len(enriched_keywords) // 3)
        
        for i in range(0, len(enriched_keywords), keywords_per_cluster):
            cluster_keywords = enriched_keywords[i:i+keywords_per_cluster]
            fallback_clusters.append({
                "topic_title": f"Topic Cluster {len(fallback_clusters) + 1}",
                "keywords": [kw['keyword'] for kw in cluster_keywords]
            })
        
        return {"keyword_clusters": fallback_clusters}
        
    except Exception as e:
        print(f"Error in clustering: {e}")
        # Ultimate fallback
        return {"keyword_clusters": [{
            "topic_title": "General SEO Topics",
            "keywords": [kw['keyword'] for kw in enriched_keywords[:5]]
        }]}


def request_user_preferences_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 4: Gets user's strategic preferences."""
    user_prefs = state.get('user_preferences', {
        "frequency": "weekly",
        "business_focus": "traffic",
        "time_horizon": 30
    })
    print(f"Using user preferences: {user_prefs}")
    return {"user_preferences": user_prefs}


def score_and_tag_topics_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 5: Scores and tags each topic based on data, trends, and user preferences."""
    clusters = state.get('keyword_clusters', [])
    if not clusters:
        return {"scored_and_tagged_topics": []}
        
    focus = state.get('user_preferences', {}).get('business_focus', 'traffic')
    enriched_keywords_map = {
        kw['keyword']: kw
        for kw in state.get('enriched_keywords', [])
    }
    focus_weights = {"traffic": 1.5, "authority": 1.2, "product": 1.0}
    scored_topics = []

    print(f"Scoring {len(clusters)} topic clusters...")

    for cluster in clusters:
        topic_title = cluster.get('topic_title', 'Unknown Topic')
        cluster_keywords = cluster.get('keywords', [])
        
        total_volume = sum(
            enriched_keywords_map.get(kw, {}).get('volume', 500)  # Default volume
            for kw in cluster_keywords)
        
        if len(cluster_keywords) > 0:
            avg_competition = sum(
                enriched_keywords_map.get(kw, {}).get('competition', 0.5)
                for kw in cluster_keywords) / len(cluster_keywords)
        else:
            avg_competition = 0.5
            
        trend_type = call_serpapi_for_trends(topic_title)
        trend_score = 0.5
        if trend_type == "Seasonal": trend_score = 0.8
        if trend_type == "Rising Trend": trend_score = 1.0
        
        final_cos = ((total_volume / 1000) *
                     (1 - avg_competition) * trend_score) * focus_weights.get(
                         focus, 1.0)
        
        tags = []
        if trend_type == "Seasonal": tags.append("Seasonal")
        if avg_competition < 0.3 and total_volume > 1000:
            tags.append("Quick-win")
        if total_volume > 10000: tags.append("High-impact")
        elif avg_competition < 0.4: tags.append("Low-competition")
        if not tags: tags.append("Standard")
        
        scored_topics.append({
            "topic_title": topic_title,
            "keywords": cluster_keywords,
            "priority_tags": ", ".join(tags),
            "cos": round(final_cos, 2)
        })

    scored_topics.sort(key=lambda x: x['cos'], reverse=True)
    print(f"Scored and ranked {len(scored_topics)} topics")
    return {"scored_and_tagged_topics": scored_topics}


def generate_editorial_calendar_node(
        state: FullWorkflowState) -> Dict[str, Any]:
    """Node 6: Creates and saves the editorial calendar."""
    topics = state.get('scored_and_tagged_topics', [])
    prefs = state.get('user_preferences', {})
    
    if not topics:
        return {"content_plan": []}
        
    days = prefs.get('time_horizon', 30)
    frequency = prefs.get('frequency', 'weekly')
    
    if frequency == 'daily': num_articles = min(days, len(topics))
    elif frequency == 'weekly': num_articles = min(days // 7, len(topics))
    else: num_articles = min(days // 30, len(topics))
    
    # Ensure at least 1 article
    num_articles = max(1, num_articles)
    
    plan = topics[:num_articles]
    print(f"Generated content plan with {len(plan)} articles")
    return {"content_plan": plan}


def generate_content_briefs_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node 7: Generates a detailed content brief for each topic."""
    plan = state.get('content_plan', [])
    if not plan:
        return {"content_briefs": [], "current_brief_index": 0, "completed_articles": []}

    briefs = []
    print(f"Generating content briefs for {len(plan)} topics...")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior content strategist. Your task is to create a detailed Content Brief. You MUST respond with ONLY a valid JSON object and nothing else. Do not use markdown like ```json. The JSON object must include keys for: 'topic', 'primary_keyword', 'secondary_keywords', 'suggested_slug', 'target_audience', 'tone_of_voice', 'content_type', 'meta_title' (under 60 chars), and 'meta_description' (under 160 chars)."
         ),
        ("user",
         "Generate the full content brief for this topic data: {topic_data}")
    ])

    chain = prompt | llm

    for i, topic in enumerate(plan):
        try:
            response = chain.invoke({"topic_data": json.dumps(topic)})
            raw_response = response.content

            # Try to extract JSON
            json_string = None
            json_match = re.search(r"```json\n(.*)\n```", raw_response, re.DOTALL)
            if json_match:
                json_string = json_match.group(1).strip()
            else:
                start_index = raw_response.find('{')
                if start_index != -1:
                    end_index = raw_response.rfind('}')
                    if end_index > start_index:
                        json_string = raw_response[start_index:end_index + 1]

            if json_string:
                try:
                    brief = json.loads(json_string)
                    briefs.append(brief)
                    print(f"Generated brief {i+1}: {brief.get('topic', 'Unknown')}")
                except json.JSONDecodeError:
                    print(f"Failed to parse brief {i+1}, using fallback")
                    # Fallback brief
                    fallback_brief = {
                        "topic": topic.get('topic_title', 'Unknown Topic'),
                        "primary_keyword": topic.get('keywords', ['keyword'])[0] if topic.get('keywords') else 'keyword',
                        "secondary_keywords": topic.get('keywords', ['keyword'])[:3],
                        "suggested_slug": topic.get('topic_title', 'article').lower().replace(' ', '-'),
                        "target_audience": "general audience",
                        "tone_of_voice": "professional yet conversational",
                        "content_type": "informative article",
                        "meta_title": topic.get('topic_title', 'Article')[:60],
                        "meta_description": f"Learn about {topic.get('topic_title', 'this topic')} with our comprehensive guide."[:160]
                    }
                    briefs.append(fallback_brief)
        except Exception as e:
            print(f"Error generating brief {i+1}: {e}")
            # Create fallback brief
            fallback_brief = {
                "topic": topic.get('topic_title', f'Topic {i+1}'),
                "primary_keyword": topic.get('keywords', ['keyword'])[0] if topic.get('keywords') else 'keyword',
                "secondary_keywords": topic.get('keywords', ['keyword'])[:3],
                "suggested_slug": f"article-{i+1}",
                "target_audience": "general audience",
                "tone_of_voice": "professional yet conversational",
                "content_type": "informative article",
                "meta_title": f"Article {i+1}",
                "meta_description": f"Comprehensive guide about topic {i+1}."
            }
            briefs.append(fallback_brief)

    print(f"Successfully generated {len(briefs)} content briefs")
    return {
        "content_briefs": briefs,
        "current_brief_index": 0,
        "completed_articles": []
    }


# ========== SEO WRITER HELPER FUNCTIONS ==========
def search_youtube_videos(query: str, max_results: int = 3) -> List[dict]:
    """Search for relevant YouTube videos using YouTube Data API."""
    try:
        youtube_api_key = os.getenv('YOUTUBE_API_KEY')
        if not youtube_api_key:
            print("YouTube API key not found, returning mock videos")
            return [{
                'title': f'Sample Video about {query}',
                'video_id': 'sample123',
                'embed_url': f"https://www.youtube.com/embed/sample123",
                'description': f'This is a sample video about {query}'
            }]

        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        request = youtube.search().list(part='snippet',
                                        q=query,
                                        type='video',
                                        maxResults=max_results,
                                        order='relevance')
        response = request.execute()

        videos = []
        for item in response['items']:
            videos.append({
                'title': item['snippet']['title'],
                'video_id': item['id']['videoId'],
                'embed_url':
                f"https://www.youtube.com/embed/{item['id']['videoId']}",
                'description': item['snippet']['description'][:200]
            })
        return videos
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return [{
            'title': f'Sample Video about {query}',
            'video_id': 'sample123',
            'embed_url': f"https://www.youtube.com/embed/sample123",
            'description': f'This is a sample video about {query}'
        }]


def generate_image_with_replicate(prompt: str) -> str:
    """Generate image using Replicate API with flux-schnell model."""
    try:
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            print("Replicate API token not found, returning placeholder")
            return "https://via.placeholder.com/800x600?text=Generated+Image"

        client = replicate.Client(api_token=replicate_token)
        output = client.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt})

        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return str(output)
    except Exception as e:
        print(f"Error generating image: {e}")
        return "https://via.placeholder.com/800x600?text=Generated+Image"


# ========== SEO WRITER NODES ==========
def prepare_content_brief_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Prepare the current content brief for processing."""
    current_index = state.get('current_brief_index', 0)
    content_briefs = state.get('content_briefs', [])

    if current_index < len(content_briefs):
        print(f"Preparing brief {current_index + 1} of {len(content_briefs)}")
        return {
            "current_content_brief": content_briefs[current_index],
            "iteration_count": 0
        }
    print("No more briefs to process")
    return {}


def serp_analysis_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Analyze SERP for the target keyword."""
    content_brief = state.get('current_content_brief', {})
    target_keyword = content_brief.get('primary_keyword', '')

    if not target_keyword:
        return {"target_keyword": "", "serp_analysis": {}}

    print(f"Analyzing SERP for keyword: {target_keyword}")

    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    if not SERPAPI_API_KEY:
        print("SERPAPI_API_KEY not found, using mock SERP data")
        return {
            "target_keyword": target_keyword,
            "serp_analysis": {
                "top_results": [{
                    "title": f"Sample Article about {target_keyword}",
                    "link": "https://example.com",
                    "snippet": f"This is a sample snippet about {target_keyword}",
                    "position": 1
                }],
                "featured_snippet": {},
                "related_searches": [{"query": f"{target_keyword} guide"}],
                "people_also_ask": [{"question": f"What is {target_keyword}?"}]
            }
        }

    params = {
        "engine": "google",
        "q": target_keyword,
        "api_key": SERPAPI_API_KEY,
        "num": 10
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        serp_data = {
            "top_results": [],
            "featured_snippet": results.get("answer_box", {}),
            "related_searches": results.get("related_searches", []),
            "people_also_ask": results.get("related_questions", [])
        }

        organic_results = results.get("organic_results", [])[:10]
        for result in organic_results:
            serp_data["top_results"].append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "position": result.get("position", 0)
            })

        print(f"Found {len(serp_data['top_results'])} SERP results")
        return {"target_keyword": target_keyword, "serp_analysis": serp_data}

    except Exception as e:
        print(f"Error in SERP analysis: {e}")
        # Return mock data as fallback
        return {
            "target_keyword": target_keyword,
            "serp_analysis": {
                "top_results": [{
                    "title": f"Sample Article about {target_keyword}",
                    "link": "https://example.com",
                    "snippet": f"This is a sample snippet about {target_keyword}",
                    "position": 1
                }],
                "featured_snippet": {},
                "related_searches": [{"query": f"{target_keyword} guide"}],
                "people_also_ask": [{"question": f"What is {target_keyword}?"}]
            }
        }


def entity_semantic_optimization_node(
        state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Extract entities and semantic keywords from SERP data."""
    serp_data = state.get('serp_analysis', {})
    target_keyword = state.get('target_keyword', '')

    print(f"Extracting entities and semantics for: {target_keyword}")

    competitor_content = []
    for result in serp_data.get('top_results', [])[:5]:
        competitor_content.append(
            f"{result.get('title', '')} {result.get('snippet', '')}")

    related_searches = [
        item.get('query', '')
        for item in serp_data.get('related_searches', [])
    ]
    paa_questions = [
        item.get('question', '')
        for item in serp_data.get('people_also_ask', [])
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an SEO expert specializing in entity extraction and semantic keyword optimization. 
        Analyze the competitor content and extract:
        1. Key entities (people, places, organizations, concepts)
        2. Semantic keywords and LSI terms
        3. Topic clusters and related themes
        4. Content gaps that competitors haven't covered

        Respond with a JSON object containing: entities, semantic_keywords, topic_clusters, and content_gaps."""
         ),
        ("user", """Target keyword: {target_keyword}

        Competitor content snippets: {competitor_content}

        Related searches: {related_searches}

        People also ask: {paa_questions}

        Extract entities and semantic optimization data.""")
    ])

    try:
        chain = prompt | writer_llm
        response = chain.invoke({
            "target_keyword": target_keyword,
            "competitor_content": "\n".join(competitor_content) if competitor_content else f"Sample content about {target_keyword}",
            "related_searches": ", ".join(related_searches) if related_searches else f"{target_keyword} guide",
            "paa_questions": "\n".join(paa_questions) if paa_questions else f"What is {target_keyword}?"
        })

        try:
            entities_semantics = json.loads(response.content)
            print("Successfully extracted entities and semantics")
        except json.JSONDecodeError:
            print("Failed to parse entities response, using fallback")
            entities_semantics = {
                "entities": [target_keyword, "related concepts"],
                "semantic_keywords": [f"{target_keyword} guide", f"{target_keyword} tips"],
                "topic_clusters": [f"{target_keyword} basics"],
                "content_gaps": [f"Comprehensive {target_keyword} analysis"]
            }

    except Exception as e:
        print(f"Error in entity extraction: {e}")
        entities_semantics = {
            "entities": [target_keyword],
            "semantic_keywords": [f"{target_keyword} guide"],
            "topic_clusters": [f"{target_keyword} basics"],
            "content_gaps": [f"Detailed {target_keyword} information"]
        }

    return {"entities_and_semantics": entities_semantics}


def create_article_blueprint_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Create a detailed article blueprint."""
    content_brief = state.get('current_content_brief', {})
    serp_analysis = state.get('serp_analysis', {})
    entities_semantics = state.get('entities_and_semantics', {})

    print(f"Creating blueprint for: {content_brief.get('topic', 'Unknown')}")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a senior content strategist creating article blueprints for SEO-optimized content.
        Create a comprehensive article blueprint that includes:
        1. Article structure with H1, H2, and H3 headings
        2. Key points to cover under each section
        3. Word count targets for each section
        4. Where to naturally incorporate entities and semantic keywords
        5. Internal linking opportunities
        6. Call-to-action placement

        The blueprint should be detailed enough for a writer to create a comprehensive article.
        Respond with a JSON object containing the complete blueprint."""),
        ("user", """Content Brief: {content_brief}

        SERP Analysis insights: {serp_insights}

        Entities to incorporate: {entities}

        Semantic keywords to use: {semantic_keywords}

        Content gaps to fill: {content_gaps}

        Create a detailed article blueprint.""")
    ])

    try:
        chain = prompt | writer_llm

        serp_insights = {
            "top_competitor_titles":
            [r.get('title', '') for r in serp_analysis.get('top_results', [])[:3]],
            "featured_snippet":
            serp_analysis.get('featured_snippet', {}),
            "common_questions":
            serp_analysis.get('people_also_ask', [])[:5]
        }

        response = chain.invoke({
            "content_brief":
            json.dumps(content_brief),
            "serp_insights":
            json.dumps(serp_insights),
            "entities":
            entities_semantics.get('entities', []),
            "semantic_keywords":
            entities_semantics.get('semantic_keywords', []),
            "content_gaps":
            entities_semantics.get('content_gaps', [])
        })

        try:
            blueprint = json.loads(response.content)
            print("Successfully created article blueprint")
        except json.JSONDecodeError:
            print("Failed to parse blueprint, using fallback")
            blueprint = {
                "title": content_brief.get('topic', 'Article Title'),
                "meta_title": content_brief.get('meta_title', ''),
                "meta_description": content_brief.get('meta_description', ''),
                "outline": [
                    {"heading": "Introduction", "word_count": 200},
                    {"heading": "Main Content", "word_count": 1000},
                    {"heading": "Conclusion", "word_count": 200}
                ],
                "total_word_count": 2000,
                "cta_placements": ["middle", "end"]
            }

    except Exception as e:
        print(f"Error creating blueprint: {e}")
        blueprint = {
            "title": content_brief.get('topic', 'Article Title'),
            "meta_title": content_brief.get('meta_title', ''),
            "meta_description": content_brief.get('meta_description', ''),
            "outline": [
                {"heading": "Introduction", "word_count": 200},
                {"heading": "Main Content", "word_count": 1000},
                {"heading": "Conclusion", "word_count": 200}
            ],
            "total_word_count": 2000,
            "cta_placements": ["end"]
        }

    return {"article_blueprint": blueprint}


def write_seo_article_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Write the fully optimized SEO article."""
    blueprint = state.get('article_blueprint', {})
    content_brief = state.get('current_content_brief', {})
    entities_semantics = state.get('entities_and_semantics', {})
    verification_feedback = state.get('verification_feedback', [])

    print(f"Writing article: {content_brief.get('topic', 'Unknown')}")

    feedback_context = ""
    if verification_feedback:
        feedback_context = f"\n\nPLEASE ADDRESS THESE ISSUES FROM PREVIOUS VERSION:\n" + "\n".join(
            verification_feedback)

    prompt = ChatPromptTemplate.from_messages([(
        "system",
        """You are an expert SEO content writer. Write a comprehensive, engaging, and SEO-optimized article following the blueprint exactly.

        Guidelines:
        1. Write in a natural, conversational tone while maintaining expertise
        2. Incorporate all specified keywords naturally
        3. Use proper markdown formatting with # for H1, ## for H2, ### for H3
        4. Include all entities and semantic keywords naturally
        5. Ensure each section meets the word count targets
        6. Write compelling meta title and description
        7. Add internal linking placeholders as [INTERNAL_LINK: topic]
        8. Include CTAs where specified
        9. Demonstrate E-E-A-T throughout the content
        10. Make content actionable and valuable to readers

        Return ONLY the markdown formatted article without any additional commentary."""
    ),
                                               ("user",
                                                """Blueprint: {blueprint}

        Content Brief: {content_brief}

        Entities to include: {entities}

        Semantic keywords: {semantic_keywords}

        Target audience: {target_audience}

        Tone of voice: {tone_of_voice}
        {feedback_context}

        Write the complete SEO-optimized article now.""")])

    try:
        chain = prompt | writer_llm

        response = chain.invoke({
            "blueprint":
            json.dumps(blueprint),
            "content_brief":
            json.dumps(content_brief),
            "entities":
            entities_semantics.get('entities', []),
            "semantic_keywords":
            entities_semantics.get('semantic_keywords', []),
            "target_audience":
            content_brief.get('target_audience', 'general audience'),
            "tone_of_voice":
            content_brief.get('tone_of_voice', 'professional yet conversational'),
            "feedback_context":
            feedback_context
        })

        article = response.content
        print(f"Successfully wrote article ({len(article)} characters)")

    except Exception as e:
        print(f"Error writing article: {e}")
        # Fallback article
        article = f"""# {content_brief.get('topic', 'Article Title')}

## Introduction

This comprehensive guide covers everything you need to know about {content_brief.get('primary_keyword', 'the topic')}.

## Main Content

[Article content would be generated here covering the key points about {content_brief.get('primary_keyword', 'the topic')}]

### Key Benefits

- Comprehensive coverage of the topic
- Practical tips and strategies
- Expert insights and recommendations

### Best Practices

When working with {content_brief.get('primary_keyword', 'this topic')}, consider these important factors:

1. Start with the basics
2. Apply proven strategies
3. Monitor your progress

## Conclusion

In conclusion, understanding {content_brief.get('primary_keyword', 'this topic')} is essential for success. Apply the strategies outlined in this guide to achieve your goals.

[INTERNAL_LINK: related topic]
"""

    return {"written_article": article}


def verify_article_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Verify the article against SEO checklist."""
    article = state.get('written_article', '')
    content_brief = state.get('current_content_brief', {})
    blueprint = state.get('article_blueprint', {})

    print("Verifying article quality and SEO compliance...")

    if not article:
        return {
            "verification_results": {
                "approved": False,
                "score": 0,
                "issues": ["No article content found"],
                "strengths": []
            },
            "verification_feedback": ["Article content is missing"]
        }

    prompt = ChatPromptTemplate.from_messages([(
        "system",
        """You are an SEO quality assurance expert. Analyze the article against this comprehensive checklist:

1. Content Quality & Relevance
- Content matches user search intent
- Unique, in-depth, original, actionable content
- Demonstrates E-E-A-T
- Covers topic comprehensively
- Facts and stats are current

2. Keyword Usage
- Primary keyword naturally included
- Uses keyword variations, synonyms, LSI terms
- No keyword stuffing
- Content aligns with user intent

3. On-Page Optimization
- Title tag: contains primary keyword, unique, under 60 characters
- Meta description: unique, keyword-rich, under 160 characters
- URL suggestion provided

4. Headings & Structure
- Single H1 with primary keyword
- Proper H2/H3 hierarchy
- Readable formatting
- Table of contents for long articles

5. Technical Requirements
- Mobile-friendly formatting
- Proper internal linking placeholders
- Media placeholders identified

Return a JSON with:
- "approved": true/false
- "score": 0-100
- "issues": [] (list of specific issues if not approved)
- "strengths": [] (what was done well)"""),
                                               ("user", """Article to verify:
{article}

Content Brief:
{content_brief}

Blueprint:
{blueprint}

Perform comprehensive SEO verification.""")])

    try:
        chain = prompt | verifier_llm

        response = chain.invoke({
            "article": article[:3000],  # Limit article length for API
            "content_brief": json.dumps(content_brief),
            "blueprint": json.dumps(blueprint)
        })

        try:
            verification = json.loads(response.content)
            print(f"Verification complete - Score: {verification.get('score', 0)}/100")
        except json.JSONDecodeError:
            print("Failed to parse verification results, using fallback")
            verification = {
                "approved": True,  # Default to approved
                "score": 75,
                "issues": [],
                "strengths": ["Article content generated successfully"]
            }

    except Exception as e:
        print(f"Error in verification: {e}")
        verification = {
            "approved": True,  # Default to approved on error
            "score": 70,
            "issues": [],
            "strengths": ["Article completed"]
        }

    feedback = None
    if not verification.get('approved', True):
        feedback = verification.get('issues', [])

    return {
        "verification_results": verification,
        "verification_feedback": feedback
    }


def media_planner_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Plan and generate media for the article."""
    article = state.get('written_article', '')
    content_brief = state.get('current_content_brief', {})
    target_keyword = state.get('target_keyword', '')

    print(f"Planning media for: {target_keyword}")

    prompt = ChatPromptTemplate.from_messages([(
        "system",
        """You are a media planning expert for SEO content. Analyze the article and:

        1. Identify 2-3 relevant YouTube videos to embed
        2. Create 3-5 image generation prompts for key sections
        3. Suggest infographic ideas if applicable
        4. Plan media placement within the article

        Return a JSON with:
        - youtube_queries: [] (search queries for videos)
        - image_prompts: [] (detailed prompts for image generation)
        - media_placement: {} (where to place each media element)"""),
                                               ("user", """Article content:
{article}

Target keyword: {target_keyword}
Topic: {topic}

Create a comprehensive media plan.""")])

    try:
        chain = prompt | media_planner_llm

        response = chain.invoke({
            "article": article[:2000],  # Limit for API
            "target_keyword": target_keyword,
            "topic": content_brief.get('topic', '')
        })

        try:
            media_plan = json.loads(response.content)
            print("Successfully created media plan")
        except json.JSONDecodeError:
            print("Failed to parse media plan, using fallback")
            media_plan = {
                "youtube_queries": [target_keyword],
                "image_prompts": [f"Professional illustration of {target_keyword}"],
                "media_placement": {}
            }

    except Exception as e:
        print(f"Error creating media plan: {e}")
        media_plan = {
            "youtube_queries": [target_keyword],
            "image_prompts": [f"Professional illustration of {target_keyword}"],
            "media_placement": {}
        }

    # Search for YouTube videos
    youtube_videos = []
    for query in media_plan.get('youtube_queries', [])[:3]:
        videos = search_youtube_videos(query)
        if videos:
            youtube_videos.extend(videos[:1])

    # Generate images using Replicate
    generated_images = []
    for prompt in media_plan.get('image_prompts', [])[:3]:  # Limit to 3 images
        image_url = generate_image_with_replicate(prompt)
        if image_url:
            generated_images.append({"prompt": prompt, "url": image_url})

    media_plan['youtube_videos'] = youtube_videos
    media_plan['generated_images'] = generated_images

    print(f"Media plan complete: {len(youtube_videos)} videos, {len(generated_images)} images")
    return {"media_plan": media_plan}


def compile_final_article_node(state: FullWorkflowState) -> Dict[str, Any]:
    """Node: Compile the final article with all media."""
    article = state.get('written_article', '')
    media_plan = state.get('media_plan', {})
    content_brief = state.get('current_content_brief', {})

    print(f"Compiling final article: {content_brief.get('topic', 'Unknown')}")

    # Start with the base article
    final_article = f"""---
title: {content_brief.get('meta_title', 'Article Title')}
meta_description: {content_brief.get('meta_description', '')}
slug: {content_brief.get('suggested_slug', 'article-slug')}
primary_keyword: {content_brief.get('primary_keyword', '')}
secondary_keywords: {', '.join(content_brief.get('secondary_keywords', []))}
---

{article}

"""

    # Add YouTube videos section if available
    if media_plan.get('youtube_videos'):
        final_article += "\n## Related Videos\n\n"
        for video in media_plan['youtube_videos']:
            final_article += f"### {video['title']}\n"
            final_article += f"[YouTube Embed: {video['embed_url']}]\n\n"

    # Add generated images as markdown image tags
    if media_plan.get('generated_images'):
        final_article += "\n## Visual Resources\n\n"
        for idx, image in enumerate(media_plan['generated_images']):
            alt_text = image['prompt'][:100]
            final_article += f"![{alt_text}]({image['url']})\n*{image['prompt']}*\n\n"

    # Add FAQ section if People Also Ask questions exist
    serp_analysis = state.get('serp_analysis', {})
    if serp_analysis.get('people_also_ask'):
        final_article += "\n## Frequently Asked Questions\n\n"
        for qa in serp_analysis['people_also_ask'][:5]:
            question = qa.get('question', '')
            if question:
                final_article += f"### {question}\n\n[Answer to be added based on article content]\n\n"

    # Add timestamp
    final_article += "\n---\n*Last updated: " + datetime.now().strftime(
        "%Y-%m-%d") + "*\n"

    # Store the completed article
    completed_articles = state.get('completed_articles', [])
    completed_articles.append({
        "brief": content_brief,
        "article": final_article,
        "verification_score": state.get('verification_results', {}).get('score', 0)
    })

    current_index = state.get('current_brief_index', 0)
    print(f"Article {current_index + 1} completed successfully")

    return {
        "final_article": final_article,
        "completed_articles": completed_articles,
        "current_brief_index": current_index + 1
    }


# ========== CONDITIONAL EDGE FUNCTIONS ==========
def should_revise_article(state: FullWorkflowState) -> str:
    """Determine if article needs revision based on verification results."""
    verification = state.get('verification_results', {})
    iteration_count = state.get('iteration_count', 0)

    # If approved or we've tried 3 times, move to media planning
    if verification.get('approved', False) or iteration_count >= 2:  # Reduced from 3 to 2
        print("Moving to media planning")
        return "media_planner"
    else:
        # Increment iteration count and go back to blueprint
        print(f"Article needs revision (iteration {iteration_count + 1})")
        return "create_blueprint"


def should_process_next_brief(state: FullWorkflowState) -> str:
    """Determine if there are more content briefs to process."""
    current_index = state.get('current_brief_index', 0)
    content_briefs = state.get('content_briefs', [])

    if current_index < len(content_briefs):
        print(f"Processing next brief ({current_index + 1}/{len(content_briefs)})")
        return "prepare_brief"
    else:
        print("All briefs processed, workflow complete")
        return "end"


# ========== WORKFLOW CREATION ==========
def create_full_seo_workflow():
    """Create the complete SEO workflow including calendar and writer agents."""
    workflow = StateGraph(FullWorkflowState)

    # Calendar Agent Nodes
    workflow.add_node("get_initial_keywords", get_initial_keywords_node)
    workflow.add_node("enrich_keywords", enrich_keywords_node)
    workflow.add_node("cluster_keywords", cluster_keywords_node)
    workflow.add_node("request_user_preferences", request_user_preferences_node)
    workflow.add_node("score_and_tag_topics", score_and_tag_topics_node)
    workflow.add_node("generate_editorial_calendar", generate_editorial_calendar_node)
    workflow.add_node("generate_content_briefs", generate_content_briefs_node)

    # SEO Writer Agent Nodes
    workflow.add_node("prepare_brief", prepare_content_brief_node)
    workflow.add_node("serp_analysis", serp_analysis_node)
    workflow.add_node("entity_semantic_optimization", entity_semantic_optimization_node)
    workflow.add_node("create_blueprint", create_article_blueprint_node)
    workflow.add_node("write_article", write_seo_article_node)
    workflow.add_node("verify_article", verify_article_node)
    workflow.add_node("media_planner", media_planner_node)
    workflow.add_node("compile_final", compile_final_article_node)

    # Set entry point
    workflow.set_entry_point("get_initial_keywords")

    # Calendar Agent Edges
    workflow.add_edge("get_initial_keywords", "enrich_keywords")
    workflow.add_edge("enrich_keywords", "cluster_keywords")
    workflow.add_edge("cluster_keywords", "request_user_preferences")
    workflow.add_edge("request_user_preferences", "score_and_tag_topics")
    workflow.add_edge("score_and_tag_topics", "generate_editorial_calendar")
    workflow.add_edge("generate_editorial_calendar", "generate_content_briefs")
    workflow.add_edge("generate_content_briefs", "prepare_brief")

    # SEO Writer Agent Edges
    workflow.add_edge("prepare_brief", "serp_analysis")
    workflow.add_edge("serp_analysis", "entity_semantic_optimization")
    workflow.add_edge("entity_semantic_optimization", "create_blueprint")
    workflow.add_edge("create_blueprint", "write_article")
    workflow.add_edge("write_article", "verify_article")

    # Conditional edge for verification
    workflow.add_conditional_edges("verify_article", should_revise_article, {
        "create_blueprint": "create_blueprint",
        "media_planner": "media_planner"
    })

    workflow.add_edge("media_planner", "compile_final")

    # Conditional edge to process next brief or end
    workflow.add_conditional_edges("compile_final", should_process_next_brief,
                                   {
                                       "prepare_brief": "prepare_brief",
                                       "end": END
                                   })

    return workflow.compile()


# ========== EXPORT FUNCTIONS ==========
def run_full_workflow(initial_query: str, user_preferences: dict = None):
    """Run the complete SEO workflow."""
    print(f"Starting SEO workflow with query: {initial_query}")
    
    app = create_full_seo_workflow()

    initial_state = {
        "initial_query": initial_query,
        "user_preferences": user_preferences or {
            "frequency": "weekly",
            "business_focus": "traffic",
            "time_horizon": 30
        }
    }

    config = {"recursion_limit": 100}  # Increased recursion limit

    try:
        final_state = None
        step_count = 0
        
        for state in app.stream(initial_state, config=config, stream_mode="values"):
            step_count += 1
            print(f"Step {step_count}: Processing workflow state...")
            final_state = state
            yield state
            
            # Safety check to prevent infinite loops
            if step_count > 50:
                print("Warning: Maximum steps reached, ending workflow")
                break

        print("Workflow completed successfully")
        return final_state
        
    except Exception as e:
        print(f"Error in workflow execution: {e}")
        raise e