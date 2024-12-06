import streamlit as st
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import nltk
import plotly.express as px

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Function to fetch comments from YouTube

def comment_fetch(video_id, api_key):
    try:
        youtube = build('youtube', "v3", developerKey=api_key)
        comments = []
        next_page_token = None
        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response["items"]:
                value = {
                    "Comment": item["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
                    "Published_At": item["snippet"]["topLevelComment"]["snippet"]['publishedAt']
                }
                comments.append(value)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []
    

def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# Function to perform sentiment analysis
def sentiment_analyser(comment):
    if not isinstance(comment, str):
        return "Invalid Comment"
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(comment)
    if sentiment["compound"] >= 0.2:
        return "Positive"
    elif sentiment["compound"] <= -0.2:
        return "Negative"
    else:
        return "Neutral"

# Streamlit page configuration
st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="auto",
)


# App Header
st.title("🎬 **YouTube Comment Sentiment Analysis**")
st.caption("Gain insights into the sentiment of YouTube comments")

# Sidebar for inputs
st.sidebar.header("Input Settings")
st.sidebar.info("Enter YouTube Video ID and API Key to start analyzing.")
video_id = st.sidebar.text_input("🔗 YouTube Video ID:" , help= "Enter the unique ID of the video. You can find it in the URL of the video. For example, in the URL https://www.youtube.com/watch?v=dQw4w9WgXcQ, the video ID is 'dQw4w9WgXcQ'.")
api_key = st.sidebar.text_input("🔑 YouTube API Key:", type="password" , help="Enter your YouTube Data API key to fetch comments from YouTube. You need to create a project in the Google Cloud Console and enable the YouTube Data API v3 to get your API key. For help, follow these steps:\n\n1. Go to the [Google Cloud Console](https://console.cloud.google.com/).\n2. Create a new project.\n3. Go to 'APIs & Services' > 'Library' and enable 'YouTube Data API v3'.\n4. Go to 'APIs & Services' > 'Credentials' and create a new API key.\n5. Paste the API key here.")




# If inputs are provided
if video_id and api_key:
    st.write("🚀 **Fetching comments... This may take a while.**")
    with st.spinner("🔄 Analyzing data..."):
        comments_data = comment_fetch(video_id, api_key)

        if comments_data:
           
            def process_comments(comments):
                df = pd.DataFrame(comments)
                sid = get_sentiment_analyzer()
                df['compound'] = df['Comment'].apply(lambda x: sid.polarity_scores(x)['compound'])
                df["Published_At"] = pd.to_datetime(df["Published_At"])
                df["Sentiment"] = df['Comment'].apply(sentiment_analyser)
                return df
            
            df = process_comments(comments_data)

            st.divider()

            # Main Section
            st.header("📊 **Analysis Results**")

            st.subheader("💬 Comments and Sentiments")
            st.dataframe(df, use_container_width=True)
            st.divider()

            st.subheader("Sentiment Distribution")
            st.markdown("Show the Distibution of Sentiment")
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            
            st.subheader("📈 Sentiment Distribution:")
            st.markdown("**Show the Percantage of Sentiment**")
            col1,col2,col3 = st.columns([1,2,1])
            fig = px.pie(df,names = "Sentiment" ,  title='Percantage of Sentiment',color_discrete_sequence= px.colors.sequential.Viridis) 
            
            fig.update_layout(
                width = 700,
                height = 700
            )
            with col2:
                st.plotly_chart(fig)

          

            # Sentiment Over Time (Line Chart)
            st.header("**Sentiment Over Time:**")
            st.markdown("**Show how Sentiment of people Changes over Time**")
            df['Date'] = pd.to_datetime(df["Published_At"])
            senti = df.groupby("Date")['compound'].mean().reset_index()
            st.line_chart(data = senti,x = 'Date',y = 'compound')
            st.divider()


            # Highlight Positive and Negative Comments
            st.header("**Highlighted Comments**")
            col3, col4 = st.columns(2)

            with col3:
                st.subheader("Top Positive Comments")
                top_positive_comments = df.sort_values(by='compound', ascending=False).head(3)
                st.table(top_positive_comments[['Comment', 'compound']])

            with col4:
                st.subheader("Top Negative Comments")
                top_negative_comments = df.sort_values(by='compound', ascending=True).head(3)
                st.table(top_negative_comments[['Comment', 'compound']])

        else:
            st.warning("⚠️ No comments fetched. Please check your Video ID or API Key.")

else:
    st.sidebar.warning("⚠️ Please enter valid inputs in the sidebar to proceed.")

# Footer
st.sidebar.markdown("© 2024 By [Ammar](https://www.instagram.com/sk_.ammar?igsh=ZmJyczE3dzlsc2d6). Built with Streamlit.", unsafe_allow_html=True)


