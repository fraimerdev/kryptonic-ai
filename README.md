# 🤖 Kryptonic AI Assistant

> Your intelligent cryptocurrency companion

Kryptonic AI Assistant is an intelligent cryptocurrency companion designed to provide real-time data, market analysis, and educational guidance. Built with Python, Flask, and the LangChain framework, it leverages Large Language Models to answer complex questions about blockchain technology, trading strategies, and the DeFi landscape.

## Key Features

* 📈 **Real-time Crypto Prices:** Get up-to-the-minute prices and market data for a wide range of cryptocurrencies.  
* 🔥 **Trending Cryptocurrencies:** Discover which coins are currently trending based on market activity and social sentiment.  
* 📊 **Market Analysis and Insights:** Receive AI-generated analysis on market trends and potential opportunities.  
* 🔗 **Blockchain Technology Questions:** Ask complex questions about how different blockchain technologies work.  
* 💡 **Trading Strategies and DeFi Guidance:** Explore trading strategies and get clear explanations on DeFi concepts.

## Tech Stack

* **Framework:** [Flask](https://flask.palletsprojects.com/)  
* **AI Orchestration:** [LangChain](https://www.langchain.com/)  
* **LLM Integration:** [OpenAI](https://openai.com/)  
* **Database:** [MongoDB](https://www.mongodb.com/) (with `pymongo`)  
* **Web Scraping:** [ScrapingAnt](https://scrapingant.com/)  
* **Production Server:** [Gunicorn](https://gunicorn.org/)  
* **Language:** Python 3

## Architecture - Flask & LangChain Backend

Kryptonic AI is a backend web application built with Python and Flask. It leverages the LangChain framework to create powerful applications powered by Large Language Models.

* 🧠 **AI-Powered Logic:** Utilizes the LangChain framework and OpenAI models to process complex user requests.  
* 🔗 **Dynamic Web Scraping:** Capable of fetching and processing live content from external websites using the ScrapingAnt service.  
* 💾 **Data Persistence:** Connects to a MongoDB database to store, retrieve, and manage application data.  
* 🌐 **Web Interface:** A simple, server-rendered frontend built with Flask and Jinja2 to interact with the backend.  
* 🔐 **Secure Configuration:** Manages API keys and secrets cleanly using environment variables.  
* 🚀 **Deployment-Ready:** Includes instructions for production deployment using Gunicorn.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

#### 1. Install Dependencies:

First, clone the repository and install the required Python packages. It is highly recommended to use a virtual environment.

```bash
# Clone the repository  
git clone https://github.com/fraimerdev/kryptonic-ai.git  
cd kryptonic-ai

# Create virtual environment
python -m venv venv  

# Activate virtual environment on MacOS/Linux
source venv/bin/activate

#Activate virtual environment on Windows
venv\Scripts\activate

# Install dependencies  
pip install -r requirements.txt
```

#### 2. **Set Up Environment Variables:**

Create a .env file in the root of the project and add your API keys and configuration.

```
# .env file

# OpenAI API Key  
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

# MongoDB Connection String  
MONGO_URI="YOUR_MONGODB_CONNECTION_STRING_HERE"

# ScrapingAnt API Key  
SCRAPINGANT_API_KEY="YOUR_SCRAPINGANT_API_KEY_HERE"

# Flask Secret Key  
SECRET_KEY="a_strong_random_secret_key"
```

#### 3. **Run the Development Server:**

Start the local Flask server.

```bash
python app.py
```

#### 4. **Open the Application:**

Visit http://127.0.0.1:5000 in your browser.

## Project Structure

```
kryptonic-ai/  
├── app.py              # Main Flask application, routes, and LangChain logic  
├── db.py               # MongoDB connection and helper functions  
├── requirements.txt    # Project dependencies for pip  
├── static/             # Static assets (CSS, JS, images)  
└── templates/          # Jinja2 HTML templates for the frontend  
```

## Deployment

To deploy this application to a production service like **Render**, **Heroku**, [PythonAnywhere](https://www.pythonanywhere.com/), or [DigitalOcean App Platform](https://www.digitalocean.com/products/app-platform), you will need to add the following files.

#### 1. **Add Gunicorn to requirements.txt:**

Ensure gunicorn is listed as a dependency for your production server.

```
# requirements.txt  
flask  
gunicorn  
python-dotenv  
langchain  
# ... all other dependencies
```

#### 2. **Create a `Procfile`:**

This file tells the hosting service how to start the application. Create a file named `Procfile` in the root directory.

```
web: gunicorn app:app
```

After adding these files, commit them and connect your repository to your hosting provider. Remember to set the environment variables in your provider's dashboard, not in the `.env` file.

## Available Scripts

* `python app.py` - Starts the local development server.  
* `gunicorn app:app` - Runs the app using the Gunicorn production server.

## Contributing

Contributions are welcome! This project serves as a template for building powerful LangChain-based web applications. Please fork the repository and submit a pull request with your suggested changes.

## License

This project is for educational and demonstration purposes.  