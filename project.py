import os
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import streamlit as st
import csv
from io import StringIO
import google.generativeai as genai

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
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
        
    def generate_instagram_posts(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_posts: int = 7) -> List[Dict]:
        if not self.model:
            return self._create_fallback_instagram_posts(brand_name, business_type, audience, product_service, goal, num_posts)
            
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
        1. Caption (engaging, with emojis, industry-specific language, 150-200 characters)
        2. 8-12 relevant hashtags (include: {', '.join(industry_hashtags[:5])})
        3. Strong call-to-action
        4. Best posting time recommendation
        5. Content type suggestion (photo, carousel, reel, story)
        6. Visual description (what the image should show)
        
        Format as JSON array with keys: day, caption, hashtags, cta, posting_time, content_type, visual_description
        
        Return only valid JSON without any other text or markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Clean up the response to extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
                
            return json.loads(json_str.strip())
        except Exception as e:
            st.error(f"Error generating Instagram content: {str(e)}")
            return self._create_fallback_instagram_posts(brand_name, business_type, audience, product_service, goal, num_posts)
    
    def generate_linkedin_posts(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_posts: int = 7) -> List[Dict]:
        if not self.model:
            return self._create_fallback_linkedin_posts(brand_name, business_type, audience, product_service, goal, num_posts)
            
        industry_context = self._get_industry_context(business_type)
        
        prompt = f"""
        Create {num_posts} DETAILED and PROFESSIONAL LinkedIn posts for {brand_name} - a {business_type.upper()} business.
        Brand: {brand_name}
        Business Type: {business_type} ({industry_context})
        Audience: {audience} (professional context)
        Product/Service: {product_service}
        Campaign Goal: {goal}
        
        REQUIREMENTS:
        1. Each post should be 300-500 words (LONG and DESCRIPTIVE)
        2. Include storytelling elements about the {business_type} industry
        3. Discuss industry insights, trends, challenges, opportunities
        4. Professional tone but engaging
        5. Include specific details about your {product_service}
        6. Add business insights and thought leadership
        7. Mention industry best practices and innovations
        
        For each post, provide:
        1. Professional caption (LONG, detailed, thought leadership style, 300-500 words)
        2. 5-8 professional hashtags including {business_type} and business terms
        3. Strong call-to-action
        4. Best posting time for B2B
        5. Post type (text, article share, poll, carousel)
        6. Key topic/theme of the post
        
        Format as JSON array with keys: day, caption, hashtags, cta, posting_time, post_type, topic
        
        Return only valid JSON without any other text or markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Clean up the response to extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
                
            return json.loads(json_str.strip())
        except Exception as e:
            st.error(f"Error generating LinkedIn content: {str(e)}")
            return self._create_fallback_linkedin_posts(brand_name, business_type, audience, product_service, goal, num_posts)
    
    def generate_promotional_emails(self, brand_name: str, business_type: str, audience: str, product_service: str, goal: str, num_emails: int = 3) -> List[Dict]:
        if not self.model:
            return self._create_fallback_emails(brand_name, business_type, audience, product_service, goal, num_emails)
            
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
        2. Email body (HTML format, detailed about your offerings, 300-400 words)
        3. Call-to-action button text
        4. Send time recommendation
        5. Email type (announcement, discount, newsletter, educational)
        6. Preview text (the text that appears after subject line)
        
        Format as JSON array with keys: email_num, subject, body, cta_button, send_time, email_type, preview_text
        
        Return only valid JSON without any other text or markdown formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            # Clean up the response to extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content
                
            return json.loads(json_str.strip())
        except Exception as e:
            st.error(f"Error generating email content: {str(e)}")
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
                "caption": f"🚀 {brand_name} is revolutionizing the {business_type} industry! Our {product_service} is designed specifically for {audience}. Experience the difference quality makes! ✨",
                "hashtags": industry_hashtags[:8] + [f"#{brand_name.lower().replace(' ', '')}", "#quality", "#innovation"],
                "cta": f"Discover our {business_type} solutions! Link in bio 👆",
                "posting_time": "6:00 PM - 8:00 PM (Peak engagement hours)",
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

At {brand_name}, our approach to {goal.lower()} centers on understanding that every business decision impacts our community of customers, partners, and stakeholders. We believe that success in {business_type} comes from consistent delivery of value, transparent communication, and continuous innovation.

As we continue to evolve our {product_service} offerings, we remain focused on the core principle that drove us from day one: creating solutions that genuinely improve the lives and businesses of {audience}.

What trends are you seeing in the {business_type} industry? How is your organization adapting to meet changing customer expectations?

#ThoughtLeadership #IndustryInsights""",
                "hashtags": [f"#{business_type.replace('_', '').title()}", "#business", "#innovation", "#growth", "#customerexperience", "#industry", "#leadership", "#entrepreneurship"],
                "cta": f"Learn more about our innovative {business_type} solutions. Connect with me to discuss industry trends.",
                "posting_time": "Tuesday-Thursday 9:00 AM - 11:00 AM (B2B peak hours)",
                "post_type": "text",
                "topic": f"{business_type.title()} Industry Innovation & Business Growth"
            } for i in range(num_posts)
        ]
    
    def _create_fallback_emails(self, brand_name, business_type, audience, product_service, goal, num_emails):
        return [
            {
                "email_num": i+1,
                "subject": f"Transform Your {business_type.title()} Experience!",
                "preview_text": f"Discover how {brand_name} is revolutionizing {business_type} solutions",
                "body": f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">{brand_name}</h1>
                        <p style="color: white; margin: 10px 0 0 0; font-size: 16px;">Revolutionizing {business_type.title()}</p>
                    </div>
                    
                    <div style="padding: 30px 20px;">
                        <h2 style="color: #2c3e50; font-size: 24px;">Hello {audience.title()}!</h2>
                        <p style="font-size: 16px; margin-bottom: 20px;">We're excited to share how {brand_name} is revolutionizing the {business_type} industry with our cutting-edge {product_service}!</p>
                        
                        <h3 style="color: #3498db; font-size: 20px; margin-top: 30px;">Why Choose Our {product_service}:</h3>
                        <ul style="font-size: 16px; line-height: 1.8;">
                            <li>🎯 Specifically designed for {audience}</li>
                            <li>⭐ Industry-leading quality and reliability</li>
                            <li>🤝 Comprehensive support and guidance</li>
                            <li>📈 Proven track record of success</li>
                            <li>💰 Competitive pricing with exceptional value</li>
                        </ul>
                        
                        <p style="font-size: 16px; margin: 20px 0;">Our mission is to help you achieve {goal.lower()} through innovative {business_type} solutions that deliver real, measurable results.</p>
                        
                        <div style="background-color: #f8f9fa; padding: 25px; border-radius: 8px; margin: 30px 0; border-left: 4px solid #3498db;">
                            <h4 style="margin-top: 0; color: #2c3e50; font-size: 18px;">💬 What Our Customers Say:</h4>
                            <p style="font-style: italic; font-size: 16px; margin: 0; color: #555;">"Working with {brand_name} has completely transformed how we approach {business_type}. Their {product_service} not only met but exceeded our expectations. The ROI has been incredible!"</p>
                            <p style="margin: 10px 0 0 0; font-weight: bold; color: #3498db;">- Sarah Johnson, Industry Leader</p>
                        </div>
                        
                        <div style="text-align: center; margin: 40px 0;">
                            <a href="#" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 35px; text-decoration: none; border-radius: 25px; font-weight: bold; font-size: 16px; display: inline-block; transition: transform 0.2s;">
                                🚀 Explore Our Solutions
                            </a>
                        </div>
                        
                        <p style="font-size: 16px; margin-top: 30px;">Ready to take your {business_type} experience to the next level? We're here to help you succeed every step of the way.</p>
                        
                        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;">
                            <p style="margin: 0; font-size: 16px;">Best regards,</p>
                            <p style="margin: 5px 0 0 0; font-weight: bold; color: #3498db; font-size: 16px;">The {brand_name} Team</p>
                        </div>
                    </div>
                    
                    <div style="background-color: #2c3e50; color: white; padding: 20px; text-align: center; font-size: 14px;">
                        <p style="margin: 0;">© 2024 {brand_name}. All rights reserved.</p>
                        <p style="margin: 10px 0 0 0;">You're receiving this because you're interested in {business_type} solutions.</p>
                    </div>
                </body>
                </html>
                """,
                "cta_button": "🚀 Explore Our Solutions",
                "send_time": "Tuesday-Thursday 10:00 AM (Optimal engagement time)",
                "email_type": "product_announcement"
            } for i in range(num_emails)
        ]

def export_to_csv(data, filename):
    """Export content data to CSV"""
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    return csv

def copy_to_clipboard(text):
    """Helper function to create a copy button"""
    return st.button("📋 Copy", key=f"copy_{hash(text)}")

def main():
    st.set_page_config(
        page_title="AI Marketing Content Generator",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AI Marketing Content Generator")
    st.markdown("### Generate Instagram, LinkedIn captions and email templates using Google Gemini AI")
    
    # API Key input at the top
    st.sidebar.header("🔑 API Configuration")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    st.sidebar.markdown("---")
    st.sidebar.info("ℹ️ This tool generates content only. No automatic posting is performed. You can copy and use the generated content on your platforms manually.")
    
    # Main interface tabs - simplified to just 3 tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Campaign Setup", 
        "📱 Content Generation", 
        "📊 Export & Analytics"
    ])
    
    with tab1:
        st.header("Campaign Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brand_name = st.text_input(
                "Brand/Company Name *", 
                placeholder="e.g., TechCorp, FashionHub, FoodiePlace"
            )
            
            # Business type selection
            business_types = [
                'fashion', 'food', 'tech', 'fitness', 'beauty', 'education',
                'finance', 'real_estate', 'healthcare', 'travel', 'automotive',
                'home', 'entertainment', 'pet', 'retail', 'general'
            ]
            
            business_type = st.selectbox(
                "Business Type *", 
                business_types,
                help="Select your business category for industry-specific content"
            )
            
            product_service = st.text_area(
                "Product/Service Description *", 
                placeholder="Describe what you offer (e.g., premium organic skincare products, AI-powered business analytics software, authentic Italian cuisine)",
                height=120
            )
        
        with col2:
            target_audience = st.text_input(
                "Target Audience *", 
                placeholder="e.g., young professionals, health-conscious consumers, small business owners"
            )
            
            campaign_goal = st.selectbox(
                "Campaign Goal *",
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
                st.write(f"**Recommended hashtags:** {', '.join(insights[:8])}")
                
                # Show optimal posting times based on business type
                posting_times = {
                    'fashion': 'Monday-Friday 12-1 PM, 7-9 PM',
                    'food': 'Monday-Friday 11 AM-1 PM, 5-7 PM, weekends 12-2 PM',
                    'tech': 'Tuesday-Thursday 9 AM-12 PM, 2-4 PM',
                    'fitness': 'Monday-Friday 6-8 AM, 5-7 PM',
                    'beauty': 'Monday-Friday 9 AM-11 AM, 7-9 PM',
                    'education': 'Tuesday-Thursday 8-10 AM, 2-4 PM',
                    'finance': 'Monday-Friday 8-10 AM, 1-3 PM',
                    'real_estate': 'Tuesday-Thursday 10 AM-12 PM, 6-8 PM',
                    'healthcare': 'Monday-Friday 9 AM-11 AM, 7-9 PM',
                    'travel': 'Tuesday-Sunday 9 AM-11 AM, 7-9 PM'
                }
                optimal_time = posting_times.get(business_type, 'Tuesday-Thursday 9 AM-11 AM, 7-9 PM')
                st.write(f"**Optimal posting times:** {optimal_time}")
                
            else:
                st.error("Please fill in all required fields marked with *")
    
    with tab2:
        st.header("Content Generation")
        
        if 'campaign_data' not in st.session_state:
            st.warning("⚠️ Please complete the Campaign Setup first")
        else:
            campaign_data = st.session_state.campaign_data
            
            if not gemini_api_key:
                st.error("🔑 Please configure your Google Gemini API key in the sidebar")
            else:
                content_generator = MarketingContentGenerator()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📱 Instagram Content")
                    num_instagram_posts = st.slider("Number of Instagram Posts", 1, 14, 7)
                    
                    if st.button("🎯 Generate Instagram Posts", type="primary"):
                        with st.spinner("🤖 AI is generating Instagram content..."):
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
                        st.success(f"✅ Generated {len(st.session_state.instagram_posts)} Instagram posts")
                        
                        for i, post in enumerate(st.session_state.instagram_posts):
                            with st.expander(f"📱 Instagram Post {i+1} - {post.get('day', f'Day {i+1}')}"):
                                # Caption with copy functionality
                                st.write("**Caption:**")
                                caption_text = post.get('caption', '')
                                st.text_area("", value=caption_text, height=100, key=f"ig_caption_{i}", disabled=True)
                                
                                # Hashtags with copy functionality
                                st.write("**Hashtags:**")
                                if isinstance(post.get('hashtags'), list):
                                    hashtags_str = ' '.join(post.get('hashtags', []))
                                else:
                                    hashtags_str = post.get('hashtags', '')
                                st.text_area("", value=hashtags_str, height=68, key=f"ig_hashtags_{i}", disabled=True)
                                
                                # Additional information
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write("**Call-to-Action:**")
                                    st.info(post.get('cta', 'N/A'))
                                    st.write("**Content Type:**")
                                    st.info(post.get('content_type', 'photo'))
                                
                                with col_b:
                                    st.write("**Best Posting Time:**")
                                    st.info(post.get('posting_time', '6:00 PM'))
                                
                                st.write("**Visual Description:**")
                                st.write(post.get('visual_description', 'N/A'))
                
                with col2:
                    st.subheader("💼 LinkedIn Content")
                    num_linkedin_posts = st.slider("Number of LinkedIn Posts", 1, 14, 7)
                    
                    if st.button("🎯 Generate LinkedIn Posts", type="primary"):
                        with st.spinner("🤖 AI is generating LinkedIn content..."):
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
                        st.success(f"✅ Generated {len(st.session_state.linkedin_posts)} LinkedIn posts")
                        
                        for i, post in enumerate(st.session_state.linkedin_posts):
                            with st.expander(f"💼 LinkedIn Post {i+1} - {post.get('day', f'Day {i+1}')}"):
                                # Caption
                                st.write("**Professional Caption:**")
                                caption_text = post.get('caption', '')
                                st.text_area("", value=caption_text, height=250, key=f"li_caption_{i}", disabled=True)
                                
                                # Hashtags
                                st.write("**Professional Hashtags:**")
                                if isinstance(post.get('hashtags'), list):
                                    hashtags_str = ' '.join(post.get('hashtags', []))
                                else:
                                    hashtags_str = post.get('hashtags', '')
                                st.text_area("", value=hashtags_str, height=68, key=f"li_hashtags_{i}", disabled=True)
                                
                                # Additional information
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write("**Call-to-Action:**")
                                    st.info(post.get('cta', 'N/A'))
                                    st.write("**Post Type:**")
                                    st.info(post.get('post_type', 'text'))
                                
                                with col_b:
                                    st.write("**Best Posting Time:**")
                                    st.info(post.get('posting_time', '9:00 AM'))
                                    st.write("**Topic/Theme:**")
                                    st.info(post.get('topic', 'Business Growth'))
                
                # Email Content Generation
                st.subheader("📧 Email Marketing Templates")
                
                col_email1, col_email2 = st.columns([2, 1])
                
                with col_email1:
                    num_emails = st.slider("Number of Email Templates", 1, 5, 3)
                    
                with col_email2:
                    if st.button("🎯 Generate Email Templates", type="primary"):
                        with st.spinner("🤖 AI is generating email templates..."):
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
                    st.success(f"✅ Generated {len(st.session_state.email_templates)} email templates")
                    
                    for i, email in enumerate(st.session_state.email_templates):
                        with st.expander(f"📧 Email Template {i+1} - {email.get('subject', f'Email {i+1}')}"):
                            col_email_a, col_email_b = st.columns(2)
                            
                            with col_email_a:
                                st.write("**Subject Line:**")
                                st.code(email.get('subject', ''), language=None)
                                
                                if 'preview_text' in email:
                                    st.write("**Preview Text:**")
                                    st.info(email.get('preview_text', ''))
                                
                                st.write("**Email Type:**")
                                st.info(email.get('email_type', 'promotional'))
                            
                            with col_email_b:
                                st.write("**Best Send Time:**")
                                st.info(email.get('send_time', '10:00 AM'))
                                
                                st.write("**CTA Button Text:**")
                                st.info(email.get('cta_button', 'Learn More'))
                            

                            # Show raw HTML for copying
                            st.write("**Raw HTML Code:**")
                            st.code(email.get('body', ''), language='html')
    
    with tab3:
        st.header("📊 Export & Analytics")
        
        if 'campaign_data' not in st.session_state:
            st.warning("⚠️ Please complete the Campaign Setup and generate content first")
        else:
            st.subheader("📥 Export Generated Content")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'instagram_posts' in st.session_state:
                    st.write("**📱 Instagram Posts**")
                    instagram_csv = export_to_csv(st.session_state.instagram_posts, "instagram_posts.csv")
                    st.download_button(
                        label="📱 Download Instagram CSV",
                        data=instagram_csv,
                        file_name=f"{st.session_state.campaign_data['brand_name']}_instagram_posts.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    st.info(f"📊 {len(st.session_state.instagram_posts)} Instagram posts ready")
                else:
                    st.info("Generate Instagram content first")
            
            with col2:
                if 'linkedin_posts' in st.session_state:
                    st.write("**💼 LinkedIn Posts**")
                    linkedin_csv = export_to_csv(st.session_state.linkedin_posts, "linkedin_posts.csv")
                    st.download_button(
                        label="💼 Download LinkedIn CSV",
                        data=linkedin_csv,
                        file_name=f"{st.session_state.campaign_data['brand_name']}_linkedin_posts.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    st.info(f"📊 {len(st.session_state.linkedin_posts)} LinkedIn posts ready")
                else:
                    st.info("Generate LinkedIn content first")
            
            with col3:
                if 'email_templates' in st.session_state:
                    st.write("**📧 Email Templates**")
                    email_csv = export_to_csv(st.session_state.email_templates, "email_templates.csv")
                    st.download_button(
                        label="📧 Download Email CSV",
                        data=email_csv,
                        file_name=f"{st.session_state.campaign_data['brand_name']}_email_templates.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    st.info(f"📊 {len(st.session_state.email_templates)} email templates ready")
                else:
                    st.info("Generate email content first")
            
            # Content Analysis
            st.subheader("📈 Content Analysis")
            
            analysis_data = []
            
            if 'instagram_posts' in st.session_state:
                ig_posts = st.session_state.instagram_posts
                avg_caption_length = sum(len(str(post.get('caption', ''))) for post in ig_posts) / len(ig_posts)
                hashtag_counts = []
                for post in ig_posts:
                    hashtags = post.get('hashtags', [])
                    if isinstance(hashtags, list):
                        hashtag_counts.append(len(hashtags))
                    else:
                        hashtag_counts.append(len(str(hashtags).split()))
                avg_hashtag_count = sum(hashtag_counts) / len(hashtag_counts) if hashtag_counts else 0
                
                analysis_data.append({
                    'Platform': 'Instagram',
                    'Posts Generated': len(ig_posts),
                    'Avg Caption Length': f"{avg_caption_length:.0f} characters",
                    'Avg Hashtags': f"{avg_hashtag_count:.1f} hashtags",
                    'Content Types': ', '.join(set(post.get('content_type', 'photo') for post in ig_posts))
                })
            
            if 'linkedin_posts' in st.session_state:
                li_posts = st.session_state.linkedin_posts
                avg_caption_length = sum(len(str(post.get('caption', ''))) for post in li_posts) / len(li_posts)
                hashtag_counts = []
                for post in li_posts:
                    hashtags = post.get('hashtags', [])
                    if isinstance(hashtags, list):
                        hashtag_counts.append(len(hashtags))
                    else:
                        hashtag_counts.append(len(str(hashtags).split()))
                avg_hashtag_count = sum(hashtag_counts) / len(hashtag_counts) if hashtag_counts else 0
                
                analysis_data.append({
                    'Platform': 'LinkedIn',
                    'Posts Generated': len(li_posts),
                    'Avg Caption Length': f"{avg_caption_length:.0f} characters",
                    'Avg Hashtags': f"{avg_hashtag_count:.1f} hashtags",
                    'Content Types': ', '.join(set(post.get('post_type', 'text') for post in li_posts))
                })
            
            if 'email_templates' in st.session_state:
                email_templates = st.session_state.email_templates
                avg_subject_length = sum(len(str(email.get('subject', ''))) for email in email_templates) / len(email_templates)
                
                analysis_data.append({
                    'Platform': 'Email',
                    'Posts Generated': len(email_templates),
                    'Avg Caption Length': f"{avg_subject_length:.0f} characters (subject)",
                    'Avg Hashtags': 'N/A',
                    'Content Types': ', '.join(set(email.get('email_type', 'promotional') for email in email_templates))
                })
            
            if analysis_data:
                df = pd.DataFrame(analysis_data)
                st.dataframe(df, use_container_width=True)
            
            # Best Practices & Recommendations
            st.subheader("💡 Best Practices & Recommendations")
            
            if 'campaign_data' in st.session_state:
                business_type = st.session_state.campaign_data['business_type']
                
                recommendations = {
                    'fashion': [
                        "Post outfit inspiration during lunch hours (12-1 PM) for maximum engagement",
                        "Use high-quality visuals showcasing texture and styling details",
                        "Include seasonal trends and styling tips in captions",
                        "Engage with fashion influencers and use trending fashion hashtags"
                    ],
                    'food': [
                        "Post food content around meal times (11 AM-1 PM, 5-7 PM)",
                        "Use appetizing visuals with good lighting and composition",
                        "Share behind-the-scenes content and cooking processes",
                        "Include nutritional information and dietary considerations"
                    ],
                    'tech': [
                        "Focus on B2B posting times (Tuesday-Thursday, 9 AM-12 PM)",
                        "Share case studies, tutorials, and industry insights",
                        "Use data visualizations and product demonstrations",
                        "Engage in tech communities and industry discussions"
                    ],
                    'fitness': [
                        "Post motivational content early morning (6-8 AM) and evening (5-7 PM)",
                        "Share transformation stories and workout tips",
                        "Include form demonstrations and safety tips",
                        "Build community through challenges and group activities"
                    ]
                }
                
                business_recommendations = recommendations.get(business_type, [
                    "Maintain consistent posting schedule for better engagement",
                    "Use high-quality visuals that align with your brand aesthetic",
                    "Engage authentically with your audience through comments and stories",
                    "Monitor analytics to optimize posting times and content types"
                ])
                
                for rec in business_recommendations:
                    st.write(f"• {rec}")
            
            # Expected Performance Metrics
            st.subheader("📊 Expected Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'instagram_posts' in st.session_state:
                    expected_reach = len(st.session_state.instagram_posts) * 300
                    st.metric("Expected Instagram Reach", f"{expected_reach:,}", "+15%")
            
            with col2:
                if 'linkedin_posts' in st.session_state:
                    expected_impressions = len(st.session_state.linkedin_posts) * 250
                    st.metric("Expected LinkedIn Impressions", f"{expected_impressions:,}", "+12%")
            
            with col3:
                if 'email_templates' in st.session_state:
                    expected_open_rate = "24.2%"
                    st.metric("Expected Email Open Rate", expected_open_rate, "+3.1%")
            
            with col4:
                total_content = 0
                if 'instagram_posts' in st.session_state:
                    total_content += len(st.session_state.instagram_posts)
                if 'linkedin_posts' in st.session_state:
                    total_content += len(st.session_state.linkedin_posts)
                if 'email_templates' in st.session_state:
                    total_content += len(st.session_state.email_templates)
                
                st.metric("Total Content Pieces", total_content, f"+{total_content}")
            
            # Usage Instructions
            st.subheader("📋 How to Use Generated Content")
            
            usage_instructions = """
            **Instagram:**
            1. Copy the caption and hashtags from the generated posts
            2. Create your visual content based on the visual descriptions provided
            3. Post at the recommended times for maximum engagement
            4. Use the suggested content types (photo, carousel, reel, story)
            
            **LinkedIn:**
            1. Copy the professional captions for thought leadership posts
            2. Post during B2B peak hours (Tuesday-Thursday, 9-11 AM)
            3. Engage with comments to build professional relationships
            4. Use the topics provided to establish industry expertise
            
            **Email Marketing:**
            1. Use the HTML email templates in your email marketing platform
            2. Customize the content with your specific offers and links
            3. Send emails at the recommended times for optimal open rates
            4. A/B test different subject lines for better performance
            """
            
            st.markdown(usage_instructions)
            
            st.info("📌 **Note:** Performance metrics are estimates based on industry averages. Actual results may vary based on audience engagement, content quality, and posting consistency.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import streamlit as st
        st.error(f"An error occurred: {e}")