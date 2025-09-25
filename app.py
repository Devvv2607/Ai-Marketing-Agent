import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction, AgentFinish
import csv
from io import StringIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import base64
from PIL import Image
import time
import schedule as sched
import threading
from urllib.parse import quote
import hashlib

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="AI Marketing Automation System",
    page_icon="🚀",
    layout="wide"
)

class BusinessTypeClassifier:
    """Classifies business type and provides industry-specific recommendations"""
    
    BUSINESS_TYPES = {
        'fashion': ['clothing', 'apparel', 'fashion', 'dress', 'shirt', 'pants', 'shoes', 'accessories', 'jewelry'],
        'food': ['restaurant', 'cafe', 'food', 'bakery', 'catering', 'delivery', 'recipe', 'cuisine', 'dining'],
        'tech': ['software', 'app', 'saas', 'technology', 'digital', 'platform', 'startup', 'ai', 'data'],
        'fitness': ['gym', 'fitness', 'workout', 'yoga', 'health', 'wellness', 'training', 'sports'],
        'beauty': ['cosmetics', 'skincare', 'makeup', 'beauty', 'salon', 'spa', 'haircare'],
        'education': ['course', 'training', 'education', 'learning', 'school', 'university', 'tutorial'],
        'finance': ['finance', 'investment', 'banking', 'insurance', 'accounting', 'money', 'loan'],
        'real_estate': ['property', 'real estate', 'housing', 'apartment', 'home', 'rent', 'buy'],
        'healthcare': ['medical', 'health', 'doctor', 'clinic', 'hospital', 'therapy', 'medicine'],
        'travel': ['travel', 'tourism', 'hotel', 'vacation', 'trip', 'booking', 'adventure'],
        'automotive': ['car', 'auto', 'vehicle', 'motorcycle', 'repair', 'dealer', 'automotive'],
        'home': ['furniture', 'decor', 'home', 'interior', 'renovation', 'garden', 'appliance'],
        'entertainment': ['music', 'movie', 'game', 'entertainment', 'event', 'concert', 'show'],
        'pet': ['pet', 'dog', 'cat', 'animal', 'veterinary', 'grooming', 'pet care'],
        'retail': ['shop', 'store', 'retail', 'marketplace', 'boutique', 'outlet', 'mall']
    }
    
    @classmethod
    def classify_business(cls, business_description: str) -> str:
        """Classify business type based on description"""
        business_lower = business_description.lower()
        
        scores = {}
        for business_type, keywords in cls.BUSINESS_TYPES.items():
            score = sum(1 for keyword in keywords if keyword in business_lower)
            if score > 0:
                scores[business_type] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    @classmethod
    def get_industry_hashtags(cls, business_type: str) -> List[str]:
        """Get industry-specific hashtags"""
        hashtag_map = {
            'fashion': ['#fashion', '#style', '#ootd', '#trending', '#fashionista', '#clothing', '#apparel', '#design'],
            'food': ['#food', '#foodie', '#delicious', '#restaurant', '#cuisine', '#tasty', '#chef', '#foodlover'],
            'tech': ['#tech', '#innovation', '#digital', '#startup', '#technology', '#ai', '#software', '#future'],
            'fitness': ['#fitness', '#health', '#workout', '#gym', '#wellness', '#fit', '#training', '#healthy'],
            'beauty': ['#beauty', '#skincare', '#makeup', '#cosmetics', '#glowup', '#selfcare', '#beautiful'],
            'education': ['#education', '#learning', '#knowledge', '#skills', '#training', '#study', '#growth'],
            'finance': ['#finance', '#money', '#investment', '#business', '#entrepreneur', '#financial', '#wealth'],
            'real_estate': ['#realestate', '#property', '#home', '#investment', '#realtor', '#housing'],
            'healthcare': ['#health', '#medical', '#wellness', '#care', '#doctor', '#healthy', '#medicine'],
            'travel': ['#travel', '#vacation', '#adventure', '#explore', '#wanderlust', '#trip', '#tourism'],
            'automotive': ['#cars', '#automotive', '#vehicle', '#driving', '#auto', '#mechanic', '#garage'],
            'home': ['#home', '#decor', '#interior', '#design', '#furniture', '#homedecor', '#living'],
            'entertainment': ['#entertainment', '#fun', '#music', '#event', '#show', '#party', '#enjoy'],
            'pet': ['#pets', '#dogs', '#cats', '#animals', '#petcare', '#furry', '#cute', '#petlover'],
            'retail': ['#shopping', '#retail', '#store', '#sale', '#deals', '#fashion', '#buy', '#shop'],
            'general': ['#business', '#entrepreneur', '#success', '#growth', '#innovation', '#quality', '#service']
        }
        return hashtag_map.get(business_type, hashtag_map['general'])

class MarketingContentGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.7
        )
        
    def generate_instagram_posts(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_posts: int = 7) -> List[Dict]:
        industry_context = self._get_industry_context(business_type)
        industry_hashtags = BusinessTypeClassifier.get_industry_hashtags(business_type)
        
        prompt = f"""
        Create {num_posts} Instagram posts for {brand_name} - a {business_type.upper()} business.
        Brand: {brand_name}
        Business Type: {business_type} ({industry_context})
        Audience: {audience}
        Product/Service: {product_service}
        Campaign Goal: {goal}
        
        Industry-specific requirements:
        {self._get_posting_requirements(business_type)}
        
        For each post, provide:
        1. Caption (engaging, with emojis, industry-specific language)
        2. 8-12 relevant hashtags (include: {', '.join(industry_hashtags[:5])})
        3. Strong call-to-action
        4. Best posting time recommendation
        5. Content type suggestion (photo, carousel, reel, story)
        6. Visual description (what the image should show)
        
        Format as JSON array with keys: day, caption, hashtags, cta, posting_time, content_type, visual_description
        """
        
        response = self.llm.invoke(prompt)
        try:
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            else:
                json_str = content
            return json.loads(json_str)
        except:
            return self._create_fallback_instagram_posts(brand_name, business_type, audience, product_service, goal, num_posts)
    
    def generate_promotional_emails(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_emails: int = 3) -> List[Dict]:
        industry_context = self._get_industry_context(business_type)
        
        prompt = f"""
        Create {num_emails} promotional emails for {brand_name} - a {business_type.upper()} business.
        Brand: {brand_name}
        Business Type: {business_type} ({industry_context})
        Audience: {audience}
        Product/Service: {product_service}
        Campaign Goal: {goal}
        
        Industry-specific focus: {self._get_email_focus(business_type)}
        
        For each email, provide:
        1. Subject line (compelling, under 50 chars, industry-specific)
        2. Email body (HTML format, detailed about your offerings)
        3. Call-to-action button text
        4. Send time recommendation
        5. Email type (announcement, discount, newsletter, educational)
        
        Format as JSON array with keys: email_num, subject, body, cta_button, send_time, email_type
        """
        
        response = self.llm.invoke(prompt)
        try:
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            else:
                json_str = content
            return json.loads(json_str)
        except:
            return self._create_fallback_emails(brand_name, business_type, audience, product_service, goal, num_emails)
    
    def _get_industry_context(self, business_type: str) -> str:
        contexts = {
            'fashion': 'Focus on style, trends, fabric quality, design, seasonal collections',
            'food': 'Emphasize taste, quality ingredients, dining experience, nutrition, special offers',
            'tech': 'Highlight innovation, features, user experience, scalability, ROI',
            'fitness': 'Focus on health benefits, transformation, community, motivation, results',
            'beauty': 'Emphasize self-care, confidence, quality ingredients, results, tutorials',
            'education': 'Highlight learning outcomes, skill development, career growth, expertise',
            'finance': 'Focus on security, returns, financial growth, trust, expertise',
            'real_estate': 'Emphasize location, investment potential, lifestyle, market trends',
            'healthcare': 'Focus on wellness, care quality, patient outcomes, trust, expertise',
            'travel': 'Highlight experiences, destinations, adventure, memories, deals',
            'automotive': 'Focus on performance, reliability, features, maintenance, deals',
            'home': 'Emphasize comfort, style, functionality, quality, home improvement',
            'entertainment': 'Focus on fun, experiences, events, community, engagement',
            'pet': 'Emphasize pet health, happiness, care, love, community',
            'retail': 'Focus on products, deals, customer service, variety, convenience'
        }
        return contexts.get(business_type, 'Focus on quality, value, customer satisfaction, and business growth')
    
    def _get_posting_requirements(self, business_type: str) -> str:
        requirements = {
            'fashion': 'Include styling tips, outfit inspiration, fabric details, seasonal trends',
            'food': 'Show appetizing visuals, ingredients, preparation process, dining ambiance',
            'tech': 'Demonstrate features, user interfaces, problem-solving capabilities',
            'fitness': 'Include workout tips, before/after transformations, motivational content',
            'beauty': 'Show application techniques, before/after results, ingredient benefits',
            'education': 'Share learning tips, success stories, industry insights, skill development',
            'finance': 'Include market insights, financial tips, success metrics, trust signals',
            'real_estate': 'Show property features, market data, lifestyle benefits, location advantages',
            'healthcare': 'Focus on patient care, health tips, medical insights, wellness advice',
            'travel': 'Share destination highlights, travel tips, cultural experiences, deals',
            'automotive': 'Show vehicle features, performance data, maintenance tips, comparisons',
            'home': 'Display room setups, before/after renovations, design tips, product features',
            'entertainment': 'Create engaging, fun content, event highlights, behind-the-scenes',
            'pet': 'Show cute pet moments, care tips, health advice, product benefits',
            'retail': 'Display products attractively, show variety, highlight deals and quality'
        }
        return requirements.get(business_type, 'Create engaging content that showcases your unique value proposition')
    
    def _get_email_focus(self, business_type: str) -> str:
        focus_areas = {
            'fashion': 'New collections, styling guides, exclusive offers, seasonal trends',
            'food': 'Menu highlights, special offers, cooking tips, nutritional information',
            'tech': 'Product updates, feature tutorials, industry insights, case studies',
            'fitness': 'Workout plans, nutrition tips, success stories, membership offers',
            'beauty': 'Beauty tips, product tutorials, ingredient spotlights, exclusive offers',
            'education': 'Course updates, learning resources, success stories, enrollment offers',
            'finance': 'Market updates, financial tips, service benefits, consultation offers',
            'real_estate': 'Property listings, market reports, investment insights, viewing appointments',
            'healthcare': 'Health tips, service information, appointment reminders, wellness programs',
            'travel': 'Destination guides, travel deals, booking information, travel tips',
            'automotive': 'Vehicle features, maintenance reminders, special offers, industry news',
            'home': 'Design inspiration, product catalogs, home improvement tips, seasonal offers',
            'entertainment': 'Event announcements, ticket offers, behind-the-scenes content, community updates',
            'pet': 'Pet care tips, product recommendations, health advice, community stories',
            'retail': 'Product showcases, sales announcements, customer stories, shopping guides'
        }
        return focus_areas.get(business_type, 'Product/service benefits, customer value, special offers, company updates')
    
    def _create_fallback_instagram_posts(self, brand_name, business_type, audience, product_service, goal, num_posts):
        industry_hashtags = BusinessTypeClassifier.get_industry_hashtags(business_type)
        return [
            {
                "day": f"Day {i+1}",
                "caption": f"🚀 {brand_name} is revolutionizing the {business_type} industry! Our {product_service} is designed specifically for {audience}. Experience the difference quality makes! ✨ {goal}",
                "hashtags": industry_hashtags + [f"#{brand_name.lower()}", "#quality", "#innovation"],
                "cta": f"Discover our {business_type} solutions! Link in bio 👆",
                "posting_time": "6:00 PM",
                "content_type": "photo",
                "visual_description": f"High-quality image showcasing {brand_name}'s {product_service} with professional lighting and appealing composition"
            } for i in range(num_posts)
        ]
    
    def _create_fallback_emails(self, brand_name, business_type, audience, product_service, goal, num_emails):
        return [
            {
                "email_num": i+1,
                "subject": f"Transform Your {business_type.title()} Experience!",
                "body": f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <h2 style="color: #2c3e50;">Hello {business_type.title()} Enthusiast!</h2>
                    <p>We're excited to share how {brand_name} is revolutionizing the {business_type} industry!</p>
                    
                    <h3>Why Choose Our {product_service}:</h3>
                    <ul>
                        <li>Specifically designed for {audience}</li>
                        <li>Industry-leading quality and reliability</li>
                        <li>Comprehensive support and guidance</li>
                        <li>Proven track record of success</li>
                        <li>Competitive pricing with exceptional value</li>
                    </ul>
                    
                    <p>Our mission is to help you {goal} through innovative {business_type} solutions that deliver real results.</p>
                    
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                        <h4 style="margin-top: 0;">What Our Customers Say:</h4>
                        <p style="font-style: italic;">"Working with {brand_name} has transformed how we approach {business_type}. Their {product_service} exceeded our expectations!"</p>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="#" style="background-color: #3498db; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">Explore Our Solutions</a>
                    </div>
                    
                    <p>Ready to take your {business_type} experience to the next level? We're here to help you succeed.</p>
                    
                    <p>Best regards,<br>The {brand_name} Team</p>
                </body>
                </html>
                """,
                "cta_button": "Explore Our Solutions",
                "send_time": "10:00 AM",
                "email_type": "product_announcement"
            } for i in range(num_emails)
        ]

class AgenticAutomationManager:
    def __init__(self):
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('EMAIL_ADDRESS'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
        self.instagram_config = {
            'access_token': os.getenv('INSTAGRAM_ACCESS_TOKEN'),
            'account_id': os.getenv('INSTAGRAM_ACCOUNT_ID')
        }
        
        # Initialize LLM for agentic decision making
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.3
        )
        
        # Setup memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup tools for the agent
        self.tools = self._setup_agent_tools()
        self.agent = self._create_marketing_agent()
    
    def _setup_agent_tools(self):
        """Setup LangChain tools for the marketing agent"""
        
        def send_email_tool(recipient_and_content: str) -> str:
            """Send email. Input format: 'recipient@email.com|Subject|Body'"""
            try:
                parts = recipient_and_content.split('|', 2)
                if len(parts) != 3:
                    return "Error: Invalid input format. Use 'email|subject|body'"
                
                recipient, subject, body = parts
                success = self._send_email(recipient, subject, body)
                return f"Email sent successfully to {recipient}" if success else f"Failed to send email to {recipient}"
            except Exception as e:
                return f"Email sending error: {str(e)}"
        
        def post_instagram_tool(content_and_image: str) -> str:
            """Post to Instagram. Input format: 'caption|hashtags|image_description'"""
            try:
                parts = content_and_image.split('|', 2)
                if len(parts) != 3:
                    return "Error: Invalid input format. Use 'caption|hashtags|image_description'"
                
                caption, hashtags, image_desc = parts
                success = self._post_to_instagram(caption, hashtags, image_desc)
                return "Instagram post published successfully" if success else "Failed to post to Instagram"
            except Exception as e:
                return f"Instagram posting error: {str(e)}"
        
        def analyze_best_time_tool(platform_and_audience: str) -> str:
            """Analyze best posting time. Input format: 'platform|audience_type'"""
            try:
                platform, audience = platform_and_audience.split('|')
                analysis = self._analyze_optimal_timing(platform, audience)
                return f"Best posting time for {platform} targeting {audience}: {analysis}"
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        def create_hashtag_strategy_tool(business_type: str) -> str:
            """Create hashtag strategy for business type"""
            try:
                hashtags = BusinessTypeClassifier.get_industry_hashtags(business_type)
                trending = self._get_trending_hashtags(business_type)
                return f"Recommended hashtags: {', '.join(hashtags[:8])}\nTrending: {', '.join(trending[:5])}"
            except Exception as e:
                return f"Hashtag strategy error: {str(e)}"
        
        def content_optimization_tool(content_type: str) -> str:
            """Optimize content for engagement. Input: content type (instagram_post, email, etc.)"""
            try:
                optimization_tips = self._get_content_optimization_tips(content_type)
                return f"Content optimization tips for {content_type}: {optimization_tips}"
            except Exception as e:
                return f"Optimization error: {str(e)}"
        
        def audience_analysis_tool(business_type_and_audience: str) -> str:
            """Analyze target audience. Input format: 'business_type|audience_description'"""
            try:
                business_type, audience = business_type_and_audience.split('|')
                analysis = self._analyze_audience(business_type, audience)
                return f"Audience analysis for {business_type} targeting {audience}: {analysis}"
            except Exception as e:
                return f"Audience analysis error: {str(e)}"
        
        return [
            Tool(
                name="send_email",
                description="Send promotional email to a recipient. Use format: 'email|subject|body'",
                func=send_email_tool
            ),
            Tool(
                name="post_instagram",
                description="Post content to Instagram. Use format: 'caption|hashtags|image_description'",
                func=post_instagram_tool
            ),
            Tool(
                name="analyze_timing",
                description="Analyze optimal posting times for platforms. Use format: 'platform|audience_type'",
                func=analyze_best_time_tool
            ),
            Tool(
                name="hashtag_strategy",
                description="Create hashtag strategy for a business type",
                func=create_hashtag_strategy_tool
            ),
            Tool(
                name="optimize_content",
                description="Get content optimization tips for different content types",
                func=content_optimization_tool
            ),
            Tool(
                name="analyze_audience",
                description="Analyze target audience for marketing strategy. Use format: 'business_type|audience_description'",
                func=audience_analysis_tool
            )
        ]
    
    def _create_marketing_agent(self):
        """Create the marketing automation agent using LangChain"""
        
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""You are an expert AI marketing automation agent powered by LangChain. Your job is to execute comprehensive marketing campaigns across multiple platforms using available tools.

You have access to the following tools:
{tools}

Tool Names: {tool_names}

As a marketing expert, you should:
1. Analyze the marketing request and determine the best strategic approach
2. Use appropriate tools to gather insights and execute campaigns
3. Provide detailed feedback on actions taken
4. Suggest optimizations based on industry best practices
5. Consider timing, content quality, audience engagement, and ROI

When executing marketing tasks:
- Always start with audience analysis if targeting information is provided
- Optimize content before posting/sending
- Analyze optimal timing for maximum engagement
- Create relevant hashtag strategies
- Provide strategic recommendations

Be strategic, data-driven, and focus on achieving measurable marketing outcomes.

Question: {input}

Thought: Let me think about how to best approach this marketing challenge.
{agent_scratchpad}"""
        )
        
        # Create ReAct agent with LangChain
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt_template
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def execute_campaign(self, campaign_data: Dict) -> Dict:
        """Execute complete marketing campaign using LangChain AI agent"""
        try:
            campaign_prompt = f"""
            Execute a comprehensive marketing campaign with the following details:
            
            Brand: {campaign_data.get('brand_name')}
            Business Type: {campaign_data.get('business_type')}
            Target Audience: {campaign_data.get('audience')}
            Product/Service: {campaign_data.get('product_service')}
            Campaign Goal: {campaign_data.get('goal')}
            
            Tasks to execute:
            1. First, analyze the target audience for this business type
            2. Create a hashtag strategy for the business type
            3. Analyze optimal timing for Instagram posts targeting this audience
            4. If email list is provided, send a promotional email to the first recipient
            5. If Instagram is enabled, create and post engaging content
            6. Optimize all content for maximum engagement
            7. Provide a comprehensive campaign execution report with recommendations
            
            Configuration:
            - Email list: {len(campaign_data.get('email_list', []))} recipients
            - Instagram enabled: {campaign_data.get('enable_instagram', False)}
            - Auto hashtags: {campaign_data.get('auto_hashtags', True)}
            - Optimize timing: {campaign_data.get('optimize_timing', True)}
            
            Please execute these tasks step by step and provide detailed feedback on each action.
            """
            
            result = self.agent.invoke({"input": campaign_prompt})
            
            return {
                "success": True,
                "message": "Campaign executed successfully using LangChain agent",
                "details": result.get("output", "Campaign completed"),
                "intermediate_steps": result.get("intermediate_steps", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Campaign execution failed: {str(e)}",
                "details": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def chat_with_agent(self, user_query: str) -> Dict:
        """Chat with the marketing agent using LangChain"""
        try:
            result = self.agent.invoke({"input": user_query})
            return {
                "success": True,
                "response": result.get("output", "No response available"),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            return {
                "success": False,
                "response": f"Agent error: {str(e)}",
                "intermediate_steps": []
            }
    
    def _send_email(self, recipient_email: str, subject: str, body: str) -> bool:
        """Send promotional email"""
        try:
            if not all([self.email_config['email'], self.email_config['password']]):
                st.warning("Email credentials not configured")
                return False
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['email']
            msg['To'] = recipient_email
            
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['email'], self.email_config['password'])
                server.send_message(msg)
            
            return True
        except Exception as e:
            st.error(f"Email sending failed: {str(e)}")
            return False
    
    def _post_to_instagram(self, caption: str, hashtags: str, image_description: str) -> bool:
        """Post to Instagram using Graph API"""
        try:
            if not self.instagram_config['access_token']:
                st.warning("Instagram API not configured")
                return False
            
            # Create media container
            url = f"https://graph.facebook.com/v18.0/{self.instagram_config['account_id']}/media"
            
            # For demo purposes, we'll use a placeholder image service
            image_url = f"https://via.placeholder.com/1080x1080/3498db/ffffff?text={quote(image_description[:50])}"
            
            payload = {
                'image_url': image_url,
                'caption': f"{caption}\n\n{hashtags}",
                'access_token': self.instagram_config['access_token']
            }
            
            response = requests.post(url, data=payload)
            
            if response.status_code == 200:
                media_id = response.json()['id']
                
                # Publish the media
                publish_url = f"https://graph.facebook.com/v18.0/{self.instagram_config['account_id']}/media_publish"
                publish_payload = {
                    'creation_id': media_id,
                    'access_token': self.instagram_config['access_token']
                }
                
                publish_response = requests.post(publish_url, data=publish_payload)
                return publish_response.status_code == 200
            
            return False
        except Exception as e:
            st.error(f"Instagram posting failed: {str(e)}")
            return False
    
    def _analyze_optimal_timing(self, platform: str, audience_type: str) -> str:
        """Analyze optimal posting times based on platform and audience"""
        timing_data = {
            'instagram': {
                'general': 'Monday-Thursday 6 AM, 10 AM, 7-9 PM',
                'b2b': 'Tuesday-Thursday 8-10 AM, 2-4 PM',
                'fashion': 'Monday, Wednesday, Friday 12-1 PM, 7-9 PM',
                'food': 'Monday-Friday 11 AM-1 PM, 5-7 PM, weekends 12-2 PM',
                'fitness': 'Monday-Friday 6-8 AM, 5-7 PM',
                'tech': 'Tuesday-Thursday 9 AM-12 PM, 2-4 PM',
                'beauty': 'Monday-Friday 9 AM-11 AM, 6-8 PM',
                'education': 'Tuesday-Thursday 10 AM-12 PM, 3-5 PM'
            },
            'email': {
                'general': 'Tuesday-Thursday 10 AM-12 PM, 2-4 PM',
                'b2b': 'Tuesday-Thursday 10 AM-11 AM, 2-3 PM',
                'retail': 'Tuesday-Thursday 8-10 AM, Friday-Sunday 12-2 PM',
                'fashion': 'Wednesday-Friday 11 AM-1 PM, 6-8 PM',
                'food': 'Tuesday-Friday 10 AM-12 PM, 5-7 PM',
                'fitness': 'Monday-Friday 7-9 AM, 6-8 PM',
                'tech': 'Tuesday-Thursday 9 AM-11 AM, 2-4 PM'
            }
        }
        
        platform_data = timing_data.get(platform.lower(), {})
        return platform_data.get(audience_type.lower(), platform_data.get('general', 'Standard business hours'))
    
    def _get_trending_hashtags(self, business_type: str) -> List[str]:
        """Get trending hashtags for business type"""
        trending_hashtags = {
            'fashion': ['#ootdinspo', '#sustainablefashion', '#vintage', '#thrifted'],
            'food': ['#foodtrends', '#plantbased', '#homecooking', '#localfood'],
            'tech': ['#ai', '#machinelearning', '#blockchain', '#cybersecurity'],
            'fitness': ['#mindfulness', '#bodypositive', '#homeworkout', '#mentalhealth'],
            'beauty': ['#cleanbeauty', '#selfcare', '#skincareroutine', '#naturalskincare'],
            'education': ['#skillbuilding', '#onlinelearning', '#professionaldevelopment', '#upskilling'],
            'finance': ['#investing', '#cryptocurrency', '#personalfinance', '#wealthbuilding'],
            'real_estate': ['#propertyinvestment', '#homedecor', '#renovation', '#markettrends'],
            'healthcare': ['#preventivecare', '#telemedicine', '#mentalwellness', '#healthtech'],
            'travel': ['#sustainabletravel', '#digitalnomad', '#culturalexchange', '#ecotourism']
        }
        return trending_hashtags.get(business_type, ['#trending', '#viral', '#popular', '#new'])
    
    def _get_content_optimization_tips(self, content_type: str) -> str:
        """Get content optimization tips for different content types"""
        optimization_tips = {
            'instagram_post': 'Use high-quality visuals, include 8-12 relevant hashtags, post during peak hours, include clear CTA, use engaging captions with emojis',
            'email': 'Craft compelling subject lines, personalize content, include clear CTA buttons, optimize for mobile, segment your audience',
            'general': 'Focus on value delivery, maintain consistent branding, engage with your audience, track performance metrics'
        }
        return optimization_tips.get(content_type, optimization_tips['general'])
    
    def _analyze_audience(self, business_type: str, audience_description: str) -> str:
        """Analyze target audience for marketing strategy"""
        audience_insights = {
            'fashion': 'Fashion audiences respond to visual storytelling, trend-focused content, and style inspiration. Best engagement through lifestyle imagery and seasonal content.',
            'food': 'Food audiences engage with appetizing visuals, behind-the-scenes content, and recipe sharing. Peak engagement during meal times.',
            'tech': 'Tech audiences prefer educational content, product demos, and industry insights. Professional tone with clear value propositions works best.',
            'fitness': 'Fitness audiences respond to motivational content, transformation stories, and workout tips. Morning and evening posts perform well.',
            'beauty': 'Beauty audiences engage with tutorials, before/after content, and product reviews. Visual content with clear results drives engagement.',
            'education': 'Education audiences prefer valuable insights, skill development content, and success stories. Professional yet approachable tone works best.',
            'finance': 'Finance audiences respond to educational content, market insights, and trust-building materials. Professional credibility is crucial.',
            'healthcare': 'Healthcare audiences prefer informative, trustworthy content with clear health benefits. Professional tone with empathy is important.',
            'retail': 'Retail audiences engage with product showcases, deals, and customer testimonials. Visual content with clear pricing and availability works well.'
        }
        
        base_insight = audience_insights.get(business_type, 'General audiences respond to authentic, valuable content that addresses their specific needs and interests.')
        return f"{base_insight} For audience '{audience_description}', focus on personalized messaging that speaks directly to their specific pain points and aspirations."

class ScheduleManager:
    def __init__(self):
        self.scheduled_posts = []
        self.automation_manager = AgenticAutomationManager()
        
    def schedule_post(self, platform: str, content: Dict, schedule_time: datetime):
        """Schedule a post for future publishing"""
        post_job = {
            'id': hashlib.md5(f"{platform}{content}{schedule_time}".encode()).hexdigest()[:8],
            'platform': platform,
            'content': content,
            'schedule_time': schedule_time,
            'status': 'scheduled',
            'created_at': datetime.now()
        }
        
        self.scheduled_posts.append(post_job)
        
        # Schedule using the schedule library
        sched.every().day.at(schedule_time.strftime("%H:%M")).do(
            self._execute_scheduled_post, post_job
        )
        
        return post_job['id']
    
    def _execute_scheduled_post(self, post_job: Dict):
        """Execute a scheduled post"""
        try:
            platform = post_job['platform']
            content = post_job['content']
            
            if platform == 'instagram':
                success = self.automation_manager._post_to_instagram(
                    content['caption'], 
                    ' '.join(content['hashtags']), 
                    content.get('visual_description', 'Marketing post')
                )
            elif platform == 'email':
                success = self.automation_manager._send_email(
                    content['recipient'], 
                    content['subject'], 
                    content['body']
                )
            
            # Update post status
            for post in self.scheduled_posts:
                if post['id'] == post_job['id']:
                    post['status'] = 'published' if success else 'failed'
                    post['published_at'] = datetime.now()
                    break
                    
        except Exception as e:
            st.error(f"Failed to execute scheduled post: {str(e)}")
    
    def get_scheduled_posts(self) -> List[Dict]:
        """Get all scheduled posts"""
        return self.scheduled_posts
    
    def cancel_scheduled_post(self, post_id: str) -> bool:
        """Cancel a scheduled post"""
        try:
            for post in self.scheduled_posts:
                if post['id'] == post_id and post['status'] == 'scheduled':
                    post['status'] = 'cancelled'
                    return True
            return False
        except:
            return False

def main():
    try:
        st.write("App started! (Debug)")  # Debug print
        st.title("🚀 AI Marketing Automation System")
        st.markdown("### Generate and automate marketing content for Instagram and Email")

        # Sidebar for configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            st.subheader("API Keys")
            groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
            if groq_api_key:
                os.environ["GROQ_API_KEY"] = groq_api_key
            st.subheader("📧 Email Settings")
            email_address = st.text_input("Email Address", value=os.getenv("EMAIL_ADDRESS", ""))
            email_password = st.text_input("Email Password", type="password", value=os.getenv("EMAIL_PASSWORD", ""))
            if email_address and email_password:
                os.environ["EMAIL_ADDRESS"] = email_address
                os.environ["EMAIL_PASSWORD"] = email_password
            st.subheader("📱 Instagram Settings")
            instagram_token = st.text_input("Instagram Access Token", type="password", value=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""))
            instagram_account = st.text_input("Instagram Account ID", value=os.getenv("INSTAGRAM_ACCOUNT_ID", ""))
            if instagram_token:
                os.environ["INSTAGRAM_ACCESS_TOKEN"] = instagram_token
            if instagram_account:
                os.environ["INSTAGRAM_ACCOUNT_ID"] = instagram_account

        # Check for required configuration and show warnings
        missing = []
        if not os.getenv("GROQ_API_KEY"):
            missing.append("Groq API Key")
        if not os.getenv("EMAIL_ADDRESS") or not os.getenv("EMAIL_PASSWORD"):
            missing.append("Email credentials")
        if not os.getenv("INSTAGRAM_ACCESS_TOKEN") or not os.getenv("INSTAGRAM_ACCOUNT_ID"):
            missing.append("Instagram API credentials")
        if missing:
            st.warning(f"Missing configuration: {', '.join(missing)}. Please fill in the sidebar.")

        # Main interface tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Campaign Setup", 
            "📱 Content Generation", 
            "🤖 AI Agent Automation", 
            "📅 Scheduling", 
            "📊 Analytics"
        ])

        with tab1:
            st.header("Campaign Setup")
            col1, col2 = st.columns(2)
            with col1:
                brand_name = st.text_input("Brand/Company Name", placeholder="e.g., TechCorp, FashionHub, FoodiePlace")
                business_types = [
                    'fashion', 'food', 'tech', 'fitness', 'beauty', 'education',
                    'finance', 'real_estate', 'healthcare', 'travel', 'automotive',
                    'home', 'entertainment', 'pet', 'retail', 'general'
                ]
                business_type = st.selectbox(
                    "Business Type", 
                    business_types,
                    help="Select your business category for industry-specific content"
                )
                product_service = st.text_area(
                    "Product/Service Description", 
                    placeholder="Describe what you offer (e.g., premium organic skincare products, AI-powered business analytics software, authentic Italian cuisine)"
                )
            with col2:
                target_audience = st.text_input(
                    "Target Audience", 
                    placeholder="e.g., young professionals, health-conscious consumers, small business owners"
                )
                campaign_goal = st.selectbox(
                    "Campaign Goal",
                    [
                        "Brand Awareness",
                        "Lead Generation", 
                        "Sales Conversion",
                        "Customer Engagement",
                        "Product Launch",
                        "Community Building",
                        "Educational Content",
                        "Seasonal Promotion"
                    ]
                )
                campaign_duration = st.selectbox(
                    "Campaign Duration",
                    ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months"]
                )
            if st.button("💾 Save Campaign Configuration", type="primary"):
                if brand_name and business_type and product_service and target_audience:
                    st.session_state.campaign_data = {
                        'brand_name': brand_name,
                        'business_type': business_type,
                        'product_service': product_service,
                        'audience': target_audience,
                        'goal': campaign_goal,
                        'duration': campaign_duration
                    }
                    st.success("✅ Campaign configuration saved!")
                    classifier = BusinessTypeClassifier()
                    insights = classifier.get_industry_hashtags(business_type)
                    st.info(f"**Industry Insights for {business_type.title()}:**")
                    st.write(f"Recommended hashtags: {', '.join(insights[:8])}")
                else:
                    st.error("Please fill in all required fields")

        with tab2:
            st.header("Content Generation")
            if 'campaign_data' not in st.session_state:
                st.warning("Please complete the Campaign Setup first")
            else:
                campaign_data = st.session_state.campaign_data
                content_generator = MarketingContentGenerator()
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📱 Instagram Content")
                    num_instagram_posts = st.slider("Number of Instagram Posts", 1, 14, 7)
                    if st.button("Generate Instagram Posts"):
                        with st.spinner("Generating Instagram content..."):
                            try:
                                instagram_posts = content_generator.generate_instagram_posts(
                                    campaign_data['brand_name'],
                                    campaign_data['business_type'],
                                    campaign_data['audience'],
                                    campaign_data['product_service'],
                                    campaign_data['goal'],
                                    num_instagram_posts
                                )
                                st.session_state.instagram_posts = instagram_posts
                            except Exception as e:
                                st.error(f"Error generating Instagram posts: {e}")
                    if 'instagram_posts' in st.session_state:
                        st.success(f"Generated {len(st.session_state.instagram_posts)} Instagram posts")
                        for i, post in enumerate(st.session_state.instagram_posts):
                            with st.expander(f"Instagram Post {i+1} - {post.get('day', f'Day {i+1}')}"):
                                st.write("**Caption:**")
                                st.write(post.get('caption', ''))
                                st.write("**Hashtags:**")
                                st.write(' '.join(post.get('hashtags', [])))
                                st.write("**CTA:**")
                                st.write(post.get('cta', ''))
                                st.write("**Best Time:**")
                                st.write(post.get('posting_time', ''))
                                st.write("**Content Type:**")
                                st.write(post.get('content_type', ''))
                                st.write("**Visual Description:**")
                                st.write(post.get('visual_description', ''))
                with col2:
                    st.subheader("📧 Email Marketing")
                    num_emails = st.slider("Number of Email Templates", 1, 5, 3)
                    if st.button("Generate Email Templates"):
                        with st.spinner("Generating email content..."):
                            try:
                                email_templates = content_generator.generate_promotional_emails(
                                    campaign_data['brand_name'],
                                    campaign_data['business_type'],
                                    campaign_data['audience'],
                                    campaign_data['product_service'],
                                    campaign_data['goal'],
                                    num_emails
                                )
                                st.session_state.email_templates = email_templates
                            except Exception as e:
                                st.error(f"Error generating email templates: {e}")
                    if 'email_templates' in st.session_state:
                        st.success(f"Generated {len(st.session_state.email_templates)} email templates")
                        for i, email in enumerate(st.session_state.email_templates):
                            with st.expander(f"Email Template {i+1} - {email.get('subject', f'Email {i+1}')}"):
                                st.write("**Subject:**")
                                st.write(email.get('subject', ''))
                                st.write("**Email Type:**")
                                st.write(email.get('email_type', ''))
                                st.write("**Send Time:**")
                                st.write(email.get('send_time', ''))
                                st.write("**Body:**")
                                st.markdown(email.get('body', ''), unsafe_allow_html=True)
                                st.write("**CTA Button:**")
                                st.write(email.get('cta_button', ''))

        with tab3:
            st.header("🤖 LangChain AI Agent Automation")
            
            if 'campaign_data' not in st.session_state:
                st.warning("Please complete the Campaign Setup first")
            else:
                automation_manager = AgenticAutomationManager()
                
                st.subheader("Automation Settings")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    enable_instagram = st.checkbox("Auto-post to Instagram", value=False)
                    enable_email = st.checkbox("Auto-send Emails", value=False)
                
                with col2:
                    auto_hashtags = st.checkbox("Auto-generate hashtags", value=True)
                    optimize_timing = st.checkbox("Optimize posting times", value=True)
                
                with col3:
                    auto_scheduling = st.checkbox("Enable auto-scheduling", value=False)
                    send_reports = st.checkbox("Send performance reports", value=False)
                
                # Email list input for automation
                st.subheader("📧 Email List for Automation")
                email_input_method = st.radio(
                    "Email Input Method",
                    ["Manual Entry", "CSV Upload"]
                )
                
                email_list = []
                if email_input_method == "Manual Entry":
                    email_text = st.text_area(
                        "Enter email addresses (one per line)",
                        placeholder="user1@example.com\nuser2@example.com\nuser3@example.com"
                    )
                    if email_text:
                        email_list = [email.strip() for email in email_text.split('\n') if email.strip()]
                
                elif email_input_method == "CSV Upload":
                    uploaded_file = st.file_uploader("Upload CSV with email addresses", type=['csv'])
                    if uploaded_file:
                        df = pd.read_csv(uploaded_file)
                        if 'email' in df.columns:
                            email_list = df['email'].dropna().tolist()
                            st.success(f"Loaded {len(email_list)} email addresses")
                        else:
                            st.error("CSV must contain an 'email' column")
                
                # Execute Automated Campaign
                if st.button("🚀 Execute AI-Powered Campaign", type="primary"):
                    if not groq_api_key:
                        st.error("Please configure Groq API key in the sidebar")
                    elif not any([enable_instagram, enable_email]):
                        st.warning("Please enable at least one automation option")
                    else:
                        campaign_config = st.session_state.campaign_data.copy()
                        campaign_config.update({
                            'enable_instagram': enable_instagram,
                            'enable_email': enable_email,
                            'email_list': email_list,
                            'auto_hashtags': auto_hashtags,
                            'optimize_timing': optimize_timing
                        })
                        
                        with st.spinner("🤖 LangChain AI Agent is executing your marketing campaign..."):
                            result = automation_manager.execute_campaign(campaign_config)
                        
                        if result['success']:
                            st.success("✅ Campaign executed successfully by AI Agent!")
                            
                            # Show execution details
                            st.subheader("🎯 Campaign Execution Report")
                            st.write(result['details'])
                            
                            # Show intermediate steps if available
                            if result.get('intermediate_steps'):
                                st.subheader("🔍 Agent Decision Process")
                                for i, step in enumerate(result['intermediate_steps']):
                                    with st.expander(f"Step {i+1}: {step[0].tool if hasattr(step[0], 'tool') else 'Analysis'}"):
                                        st.write(f"**Action:** {step[0].tool if hasattr(step[0], 'tool') else 'Thinking'}")
                                        st.write(f"**Input:** {step[0].tool_input if hasattr(step[0], 'tool_input') else step[0]}")
                                        st.write(f"**Result:** {step[1] if len(step) > 1 else 'Processing...'}")
                            
                            # Show next steps
                            st.info("**Next Steps:**")
                            st.write("1. Monitor your Instagram account for posted content")
                            st.write("2. Check email delivery status")
                            st.write("3. Review performance analytics in the Analytics tab")
                            st.write("4. Adjust strategy based on engagement metrics")
                            
                        else:
                            st.error(f"❌ Campaign execution failed: {result['message']}")
                
                # Manual Agent Interaction
                st.subheader("💬 Chat with LangChain Marketing Agent")
                user_query = st.text_input(
                    "Ask the AI marketing agent anything:",
                    placeholder="e.g., 'What's the best time to post for fashion brands?' or 'Create a hashtag strategy for my tech startup'"
                )
                
                if st.button("Ask AI Agent") and user_query:
                    with st.spinner("AI Agent is analyzing and responding..."):
                        agent_response = automation_manager.chat_with_agent(user_query)
                        
                        if agent_response['success']:
                            st.write("**🤖 AI Agent Response:**")
                            st.write(agent_response['response'])
                            
                            # Show agent's thinking process
                            if agent_response.get('intermediate_steps'):
                                with st.expander("🧠 Agent's Thinking Process"):
                                    for i, step in enumerate(agent_response['intermediate_steps']):
                                        st.write(f"**Step {i+1}:** {step[0].tool if hasattr(step[0], 'tool') else 'Analysis'}")
                                        st.write(f"**Action:** {step[1] if len(step) > 1 else 'Processing...'}")
                        else:
                            st.error(agent_response['response'])

        with tab4:
            st.header("📅 Content Scheduling")
            
            if 'campaign_data' not in st.session_state:
                st.warning("Please complete the Campaign Setup first")
            else:
                schedule_manager = ScheduleManager()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Schedule New Content")
                    
                    platform = st.selectbox("Platform", ["Instagram", "Email"])
                    
                    # Content selection based on generated content
                    if platform == "Instagram" and 'instagram_posts' in st.session_state:
                        post_options = [f"Post {i+1}: {post.get('caption', '')[:50]}..." 
                                      for i, post in enumerate(st.session_state.instagram_posts)]
                        selected_post_idx = st.selectbox("Select Instagram Post", range(len(post_options)), 
                                                       format_func=lambda x: post_options[x])
                        selected_content = st.session_state.instagram_posts[selected_post_idx]
                    
                    elif platform == "Email" and 'email_templates' in st.session_state:
                        email_options = [f"Email {i+1}: {email.get('subject', '')}" 
                                       for i, email in enumerate(st.session_state.email_templates)]
                        selected_email_idx = st.selectbox("Select Email Template", range(len(email_options)),
                                                        format_func=lambda x: email_options[x])
                        selected_content = st.session_state.email_templates[selected_email_idx]
                        
                        # Email recipient for scheduling
                        recipient_email = st.text_input("Recipient Email (for scheduling)")
                        selected_content['recipient'] = recipient_email
                    
                    else:
                        st.info(f"Generate {platform} content first in the Content Generation tab")
                        selected_content = None
                    
                    # Schedule date and time
                    schedule_date = st.date_input("Schedule Date", min_value=datetime.now().date())
                    schedule_time = st.time_input("Schedule Time")
                    
                    schedule_datetime = datetime.combine(schedule_date, schedule_time)
                    
                    if st.button("📅 Schedule Content") and selected_content:
                        if schedule_datetime <= datetime.now():
                            st.error("Please select a future date and time")
                        else:
                            post_id = schedule_manager.schedule_post(
                                platform.lower(), 
                                selected_content, 
                                schedule_datetime
                            )
                            st.success(f"✅ Content scheduled! Post ID: {post_id}")
                
                with col2:
                    st.subheader("Scheduled Posts")
                    
                    scheduled_posts = schedule_manager.get_scheduled_posts()
                    
                    if scheduled_posts:
                        for post in scheduled_posts:
                            with st.expander(f"{post['platform'].title()} - {post['id']} ({post['status']})"):
                                st.write(f"**Scheduled for:** {post['schedule_time']}")
                                st.write(f"**Status:** {post['status']}")
                                st.write(f"**Created:** {post['created_at']}")
                                
                                if post['status'] == 'scheduled':
                                    if st.button(f"Cancel {post['id']}", key=f"cancel_{post['id']}"):
                                        if schedule_manager.cancel_scheduled_post(post['id']):
                                            st.success("Post cancelled")
                                            st.rerun()
                                        else:
                                            st.error("Failed to cancel post")
                    else:
                        st.info("No scheduled posts yet")

        with tab5:
            st.header("📊 Analytics & Performance")
            
            # Mock analytics data for demonstration
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Posts Published", "18", "+6")
            with col2:
                st.metric("Total Reach", "9.2K", "+12%")
            with col3:
                st.metric("Engagement Rate", "4.1%", "+0.8%")
            with col4:
                st.metric("Email Open Rate", "24.3%", "+1.8%")
            
            # Performance charts (mock data)
            st.subheader("Performance Over Time")
            
            # Generate sample data
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            engagement_data = pd.DataFrame({
                'Date': dates,
                'Instagram': [50 + i*2 + (i%7)*10 for i in range(30)],
                'Email': [25 + i*1 + (i%3)*5 for i in range(30)]
            })
            
            st.line_chart(engagement_data.set_index('Date'))
            
            # Platform breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Engagement by Platform")
                platform_data = pd.DataFrame({
                    'Platform': ['Instagram', 'Email'],
                    'Engagement': [450, 280]
                })
                st.bar_chart(platform_data.set_index('Platform'))
            
            with col2:
                st.subheader("Best Performing Content Types")
                content_data = pd.DataFrame({
                    'Type': ['Photos', 'Carousels', 'Reels', 'Stories'],
                    'Performance': [85, 92, 88, 75]
                })
                st.bar_chart(content_data.set_index('Type'))
            
            # AI-powered recommendations
            st.subheader("🤖 AI-Powered Recommendations")
            st.info("**Based on your performance data and LangChain analysis:**")
            recommendations = [
                "📱 Instagram carousels are performing 15% better than single photos",
                "⏰ Your audience is most active on Tuesdays and Thursdays at 7 PM",
                "📧 Email subject lines with emojis have 23% higher open rates",
                "🏷️ Posts with 8-12 hashtags get optimal reach",
                "📈 Video content generates 40% more engagement than static posts",
                "🎯 Industry-specific hashtags increase engagement by 18%"
            ]
            
            for rec in recommendations:
                st.write(f"• {rec}")

    except Exception as e:
        st.error(f"A critical error occurred: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()