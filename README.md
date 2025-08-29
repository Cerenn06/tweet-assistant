Tweet Assistant
This project is an LLM-powered content generation tool that creates summaries and persona-styled tweets from news articles. It is built with Google Gemini API and Streamlit.

🚀 Features
Extracts article text from a news URL (custom extractor with BeautifulSoup).
Generates summaries and identifies key facts.
Produces tweets in different persona styles.
Supports both single tweets and thread format (2–3 connected tweets).
Fact coverage metrics: measures how well tweets include mandatory facts.
Simple disk caching for faster repeated runs.

🔑 Requirements
Python 3.10+
Google Gemini API key (GOOGLE_API_KEY)
Streamlit

⚙️ Installation
Clone the repository or download the files
Create and activate a virtual environment
Install dependencies (requirements)
Add your API keys to the .env file

▶️ Usage
Run the Streamlit app
Enter a news URL in the input box. View the
summary and generated tweets per persona in the interface.

📌 Notes
CACHE_DIR stores cached results in .tweet_cache.
Default output language is Turkish, but prompts can be adapted to other languages.
