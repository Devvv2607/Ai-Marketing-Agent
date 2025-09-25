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
            model_name="mixtral-8x7b-32768",
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
    
    def generate_linkedin_posts(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_posts: int = 7) -> List[Dict]:
        industry_context = self._get_industry_context(business_type)
        
        prompt = f"""
        Create {num_posts} DETAILED and PROFESSIONAL LinkedIn posts for {brand_name} - a {business_type.upper()} business.
        Brand: {brand_name}
        Business Type: {business_type} ({industry_context})
        Audience: {audience} (professional context)
        Product/Service: {product_service}
        Campaign Goal: {goal}
        
        REQUIREMENTS:
        1. Each post should be 200-400 words (LONG and DESCRIPTIVE)
        2. Include storytelling elements about the {business_type} industry
        3. Discuss industry insights, trends, challenges, opportunities
        4. Professional tone but engaging
        5. Include specific details about your {product_service}
        6. Add business insights and thought leadership
        7. Mention industry best practices and innovations
        
        For each post, provide:
        1. Professional caption (LONG, detailed, thought leadership style, 200-400 words)
        2. 5-8 professional hashtags including {business_type} and business terms
        3. Strong call-to-action
        4. Best posting time for B2B
        5. Post type (text, article share, poll, carousel)
        6. Key topic/theme of the post
        
        Format as JSON array with keys: day, caption, hashtags, cta, posting_time, post_type, topic
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
            return self._create_fallback_linkedin_posts(brand_name, business_type, audience, product_service, goal, num_posts)
    
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
    
    def _create_fallback_linkedin_posts(self, brand_name, business_type, audience, product_service, goal, num_posts):
        return [
            {
                "day": f"Day {i+1}",
                "caption": f"""The {business_type} industry is experiencing unprecedented transformation, and at {brand_name}, we're at the forefront of this evolution.

Our journey in developing {product_service} for {audience} has taught us valuable lessons about innovation, customer-centricity, and market adaptation. 

Here's what we've learned:
• Understanding your target market deeply drives better product development
• Quality and consistency build lasting customer relationships  
• Innovation should solve real problems, not create complexity
• Sustainable business practices are no longer optional—they're essential
• Community building and authentic engagement create brand loyalty

The {business_type} sector is projected for significant growth, but success requires more than just following trends. It demands genuine value creation, ethical business practices, and a commitment to customer success.

At {brand_name}, our approach to {goal} centers on understanding that every business decision impacts our community of customers, partners, and stakeholders. We believe that success in {business_type} comes from consistent delivery of value, transparent communication, and continuous innovation.

As we continue to evolve our {product_service} offerings, we remain focused on the core principle that drove us from day one: creating solutions that genuinely improve the lives and businesses of {audience}.

What trends are you seeing in the {business_type} industry? How is your organization adapting to meet changing customer expectations?""",
                "hashtags": [f"#{business_type}", "#business", "#innovation", "#growth", "#customerexperience", "#industry", "#leadership", "#entrepreneurship"],
                "cta": f"Learn more about our innovative {business_type} solutions",
                "posting_time": "9:00 AM",
                "post_type": "text",
                "topic": f"{business_type.title()} Industry Innovation & Business Growth"
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
        self.linkedin_config = {
            'access_token': os.getenv('LINKEDIN_ACCESS_TOKEN'),
            'person_id': os.getenv('LINKEDIN_PERSON_ID')
        }
        
        # Initialize LLM for agentic decision making
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",
            temperature=0.3
        )
        
        # Setup tools for the agent
        self.tools = self._setup_agent_tools()
        self.agent = self._create_marketing_agent()
    
    def _setup_agent_tools(self):
        """Setup tools for the marketing agent"""
        
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
        
        def post_linkedin_tool(content: str) -> str:
            """Post to LinkedIn. Input: post content"""
            try:
                success = self._post_to_linkedin(content)
                return "LinkedIn post published successfully" if success else "Failed to post to LinkedIn"
            except Exception as e:
                return f"LinkedIn posting error: {str(e)}"
        
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
        
        return [
            Tool(name="send_email", description="Send promotional email", func=send_email_tool),
            Tool(name="post_instagram", description="Post content to Instagram", func=post_instagram_tool),
            Tool(name="post_linkedin", description="Post content to LinkedIn", func=post_linkedin_tool),
            Tool(name="analyze_timing", description="Analyze optimal posting times", func=analyze_best_time_tool),
            Tool(name="hashtag_strategy", description="Create hashtag strategy", func=create_hashtag_strategy_tool)
        ]
    
    def _create_marketing_agent(self):
        """Create the marketing automation agent"""
        prompt_template = PromptTemplate(
            input_variables=["input", "agent_scratchpad"],
            template="""You are an expert marketing automation agent. Your job is to execute marketing campaigns across multiple platforms.

Available tools:
{tools}

Tool descriptions:
{tool_names}

When executing marketing tasks:
1. Analyze the request and determine the best approach
2. Use appropriate tools to execute the campaign
3. Provide detailed feedback on actions taken
4. Suggest optimizations based on industry best practices

Always be strategic about timing, content quality, and audience engagement.

Question: {input}
{agent_scratchpad}"""
        )
        
        # Create agent with tools
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt_template
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def execute_campaign(self, campaign_data: Dict) -> Dict:
        """Execute complete marketing campaign using AI agent"""
        try:
            campaign_prompt = f"""
            Execute a comprehensive marketing campaign with the following details:
            
            Brand: {campaign_data.get('brand_name')}
            Business Type: {campaign_data.get('business_type')}
            Target Audience: {campaign_data.get('audience')}
            Product/Service: {campaign_data.get('product_service')}
            Campaign Goal: {campaign_data.get('goal')}
            
            Tasks to execute:
            1. Analyze optimal timing for this business type and audience
            2. Create and execute hashtag strategy
            3. If email list provided, send welcome/promotional email
            4. If social media is enabled, create and post content
            5. Provide campaign execution report
            
            Email list: {campaign_data.get('email_list', 'Not provided')}
            Enable Instagram: {campaign_data.get('enable_instagram', False)}
            Enable LinkedIn: {campaign_data.get('enable_linkedin', False)}
            """
            
            result = self.agent.invoke({"input": campaign_prompt})
            
            return {
                "success": True,
                "message": "Campaign executed successfully",
                "details": result.get("output", "Campaign completed"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Campaign execution failed: {str(e)}",
                "details": None,
                "timestamp": datetime.now().isoformat()
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
    
    def _post_to_linkedin(self, content: str) -> bool:
        """Post to LinkedIn using API"""
        """Post to LinkedIn using API"""
        try:
            if not self.linkedin_config['access_token']:
                st.warning("LinkedIn API not configured")
                return False
            
            url = "https://api.linkedin.com/v2/ugcPosts"
            
            headers = {
                'Authorization': f"Bearer {self.linkedin_config['access_token']}",
                'Content-Type': 'application/json',
                'X-Restli-Protocol-Version': '2.0.0'
            }
            
            payload = {
                "author": f"urn:li:person:{self.linkedin_config['person_id']}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": content
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                }
            }
            
            response = requests.post(url, headers=headers, json=payload)
            return response.status_code == 201
            
        except Exception as e:
            st.error(f"LinkedIn posting failed: {str(e)}")
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
                'tech': 'Tuesday-Thursday 9 AM-12 PM, 2-4 PM'
            },
            'linkedin': {
                'general': 'Tuesday-Thursday 8-10 AM, 12-2 PM',
                'b2b': 'Tuesday-Thursday 7:45-8:30 AM, 12-1 PM, 5-6 PM',
                'professional': 'Monday-Friday 8-10 AM, 12-2 PM, 5-6 PM'
            },
            'email': {
                'general': 'Tuesday-Thursday 10 AM-12 PM, 2-4 PM',
                'b2b': 'Tuesday-Thursday 10 AM-11 AM, 2-3 PM',
                'retail': 'Tuesday-Thursday 8-10 AM, Friday-Sunday 12-2 PM'
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
            elif platform == 'linkedin':
                success = self.automation_manager._post_to_linkedin(content['caption'])
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
    st.set_page_config(
        page_title="AI Marketing Automation System",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AI Marketing Automation System")
    st.markdown("### Generate and automate marketing content across multiple platforms")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Configuration
        st.subheader("API Keys")
        groq_api_key = st.text_input("Groq API Key", type="password", 
                                   value=os.getenv("GROQ_API_KEY", ""))
        
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Email Configuration
        st.subheader("📧 Email Settings")
        email_address = st.text_input("Email Address", value=os.getenv("EMAIL_ADDRESS", ""))
        email_password = st.text_input("Email Password", type="password", 
                                     value=os.getenv("EMAIL_PASSWORD", ""))
        
        if email_address and email_password:
            os.environ["EMAIL_ADDRESS"] = email_address
            os.environ["EMAIL_PASSWORD"] = email_password
        
        # Social Media Configuration
        st.subheader("📱 Social Media")
        instagram_token = st.text_input("Instagram Access Token", type="password", 
                                      value=os.getenv("INSTAGRAM_ACCESS_TOKEN", ""))
        instagram_account = st.text_input("Instagram Account ID", 
                                        value=os.getenv("INSTAGRAM_ACCOUNT_ID", ""))
        
        linkedin_token = st.text_input("LinkedIn Access Token", type="password", 
                                     value=os.getenv("LINKEDIN_ACCESS_TOKEN", ""))
        linkedin_person = st.text_input("LinkedIn Person ID", 
                                      value=os.getenv("LINKEDIN_PERSON_ID", ""))
        
        # Update environment variables
        if instagram_token:
            os.environ["INSTAGRAM_ACCESS_TOKEN"] = instagram_token
        if instagram_account:
            os.environ["INSTAGRAM_ACCOUNT_ID"] = instagram_account
        if linkedin_token:
            os.environ["LINKEDIN_ACCESS_TOKEN"] = linkedin_token
        if linkedin_person:
            os.environ["LINKEDIN_PERSON_ID"] = linkedin_person
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Campaign Setup", 
        "📱 Content Generation", 
        "🤖 Agentic Automation", 
        "📅 Scheduling", 
        "📊 Analytics"
    ])
    
    with tab1:
        st.header("Campaign Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brand_name = st.text_input("Brand/Company Name", placeholder="e.g., TechCorp, FashionHub, FoodiePlace")
            
            # Business type selection with comprehensive options
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
        
        # Store campaign data in session state
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
                
                # Show business insights
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
                        instagram_posts = content_generator.generate_instagram_posts(
                            campaign_data['brand_name'],
                            campaign_data['business_type'],
                            campaign_data['audience'],
                            campaign_data['product_service'],
                            campaign_data['goal'],
                            num_instagram_posts
                        )
                        st.session_state.instagram_posts = instagram_posts
                
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
                st.subheader("💼 LinkedIn Content")
                num_linkedin_posts = st.slider("Number of LinkedIn Posts", 1, 14, 7)
                
                if st.button("Generate LinkedIn Posts"):
                    with st.spinner("Generating LinkedIn content..."):
                        linkedin_posts = content_generator.generate_linkedin_posts(
                            campaign_data['brand_name'],
                            campaign_data['business_type'],
                            campaign_data['audience'],
                            campaign_data['product_service'],
                            campaign_data['goal'],
                            num_linkedin_posts
                        )
                        st.session_state.linkedin_posts = linkedin_posts
                
                if 'linkedin_posts' in st.session_state:
                    st.success(f"Generated {len(st.session_state.linkedin_posts)} LinkedIn posts")
                    
                    for i, post in enumerate(st.session_state.linkedin_posts):
                        with st.expander(f"LinkedIn Post {i+1} - {post.get('day', f'Day {i+1}')}"):
                            st.write("**Caption:**")
                            st.write(post.get('caption', ''))
                            st.write("**Hashtags:**")
                            st.write(' '.join(post.get('hashtags', [])))
                            st.write("**CTA:**")
                            st.write(post.get('cta', ''))
                            st.write("**Best Time:**")
                            st.write(post.get('posting_time', ''))
                            st.write("**Topic:**")
                            st.write(post.get('topic', ''))
            
            # Email Content Generation
            st.subheader("📧 Email Marketing")
            num_emails = st.slider("Number of Email Templates", 1, 5, 3)
            
            if st.button("Generate Email Templates"):
                with st.spinner("Generating email content..."):
                    email_templates = content_generator.generate_promotional_emails(
                        campaign_data['brand_name'],
                        campaign_data['business_type'],
                        campaign_data['audience'],
                        campaign_data['product_service'],
                        campaign_data['goal'],
                        num_emails
                    )
                    st.session_state.email_templates = email_templates
            
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
        st.header("🤖 Agentic Automation")
        
        if 'campaign_data' not in st.session_state:
            st.warning("Please complete the Campaign Setup first")
        else:
            automation_manager = AgenticAutomationManager()
            
            st.subheader("Automation Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                enable_instagram = st.checkbox("Auto-post to Instagram", value=False)
                enable_linkedin = st.checkbox("Auto-post to LinkedIn", value=False)
                enable_email = st.checkbox("Auto-send Emails", value=False)
            
            with col2:
                auto_hashtags = st.checkbox("Auto-generate hashtags", value=True)
                optimize_timing = st.checkbox("Optimize posting times", value=True)
                auto_scheduling = st.checkbox("Enable auto-scheduling", value=False)
            
            with col3:
                send_reports = st.checkbox("Send performance reports", value=False)
                auto_respond = st.checkbox("Auto-respond to comments", value=False)
                content_curation = st.checkbox("Auto-curate content", value=False)
            
            # Email list input for automation
            st.subheader("📧 Email List for Automation")
            email_input_method = st.radio(
                "Email Input Method",
                ["Manual Entry", "CSV Upload", "Connect Email Service"]
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
            if st.button("🚀 Execute Automated Campaign", type="primary"):
                if not groq_api_key:
                    st.error("Please configure Groq API key in the sidebar")
                elif not any([enable_instagram, enable_linkedin, enable_email]):
                    st.warning("Please enable at least one automation option")
                else:
                    campaign_config = st.session_state.campaign_data.copy()
                    campaign_config.update({
                        'enable_instagram': enable_instagram,
                        'enable_linkedin': enable_linkedin,
                        'enable_email': enable_email,
                        'email_list': email_list,
                        'auto_hashtags': auto_hashtags,
                        'optimize_timing': optimize_timing
                    })
                    
                    with st.spinner("🤖 AI Agent is executing your marketing campaign..."):
                        result = automation_manager.execute_campaign(campaign_config)
                    
                    if result['success']:
                        st.success("✅ Campaign executed successfully!")
                        st.json(result)
                        
                        # Show execution details
                        st.subheader("Execution Details")
                        st.write(result['details'])
                        
                        # Show next steps
                        st.info("**Next Steps:**")
                        st.write("1. Monitor your social media accounts for posted content")
                        st.write("2. Check email delivery status")
                        st.write("3. Review performance analytics in the Analytics tab")
                        st.write("4. Adjust strategy based on engagement metrics")
                        
                    else:
                        st.error(f"❌ Campaign execution failed: {result['message']}")
            
            # Manual Agent Interaction
            st.subheader("💬 Chat with Marketing Agent")
            user_query = st.text_input(
                "Ask the marketing agent anything:",
                placeholder="e.g., 'What's the best time to post for fashion brands?' or 'Create a hashtag strategy for my tech startup'"
            )
            
            if st.button("Ask Agent") and user_query:
                with st.spinner("Agent is thinking..."):
                    try:
                        agent_response = automation_manager.agent.invoke({"input": user_query})
                        st.write("**Agent Response:**")
                        st.write(agent_response.get('output', 'No response available'))
                    except Exception as e:
                        st.error(f"Agent error: {str(e)}")
    
    with tab4:
        st.header("📅 Content Scheduling")
        
        if 'campaign_data' not in st.session_state:
            st.warning("Please complete the Campaign Setup first")
        else:
            schedule_manager = ScheduleManager()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Schedule New Content")
                
                platform = st.selectbox("Platform", ["Instagram", "LinkedIn", "Email"])
                
                # Content selection based on generated content
                if platform == "Instagram" and 'instagram_posts' in st.session_state:
                    post_options = [f"Post {i+1}: {post.get('caption', '')[:50]}..." 
                                  for i, post in enumerate(st.session_state.instagram_posts)]
                    selected_post_idx = st.selectbox("Select Instagram Post", range(len(post_options)), 
                                                   format_func=lambda x: post_options[x])
                    selected_content = st.session_state.instagram_posts[selected_post_idx]
                
                elif platform == "LinkedIn" and 'linkedin_posts' in st.session_state:
                    post_options = [f"Post {i+1}: {post.get('caption', '')[:50]}..." 
                                  for i, post in enumerate(st.session_state.linkedin_posts)]
                    selected_post_idx = st.selectbox("Select LinkedIn Post", range(len(post_options)),
                                                   format_func=lambda x: post_options[x])
                    selected_content = st.session_state.linkedin_posts[selected_post_idx]
                
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
            st.metric("Posts Published", "24", "+8")
        with col2:
            st.metric("Total Reach", "12.5K", "+15%")
        with col3:
            st.metric("Engagement Rate", "3.2%", "+0.5%")
        with col4:
            st.metric("Email Open Rate", "22.8%", "+2.1%")
        
        # Performance charts (mock data)
        st.subheader("Performance Over Time")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        engagement_data = pd.DataFrame({
            'Date': dates,
            'Instagram': [50 + i*2 + (i%7)*10 for i in range(30)],
            'LinkedIn': [30 + i*1.5 + (i%5)*8 for i in range(30)],
            'Email': [25 + i*1 + (i%3)*5 for i in range(30)]
        })
        
        st.line_chart(engagement_data.set_index('Date'))
        
        # Platform breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engagement by Platform")
            platform_data = pd.DataFrame({
                'Platform': ['Instagram', 'LinkedIn', 'Email'],
                'Engagement': [450, 320, 280]
            })
            st.bar_chart(platform_data.set_index('Platform'))
        
        with col2:
            st.subheader("Best Performing Content Types")
            content_data = pd.DataFrame({
                'Type': ['Photos', 'Carousels', 'Reels', 'Stories'],
                'Performance': [85, 92, 88, 75]
            })
            st.bar_chart(content_data.set_index('Type'))
        
        # Recommendations based on analytics
        st.subheader("🎯 AI Recommendations")
        st.info("**Based on your performance data:**")
        recommendations = [
            "📱 Instagram carousels are performing 15% better than single photos",
            "⏰ Your audience is most active on Tuesdays and Thursdays at 7 PM",
            "📧 Email subject lines with emojis have 23% higher open rates",
            "🏷️ Posts with 8-12 hashtags get optimal reach",
            "📈 Video content generates 40% more engagement than static posts"
        ]
        
        for rec in recommendations:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()