import streamlit as st
import pandas as pd 
from PIL import Image

import yfinance as yf
from yahooquery import Ticker
from datetime import datetime, timedelta
from edgar import Company, TXTML 

from dotenv import load_dotenv
import os

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA 
from langchain.document_loaders import TextLoader 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI 
from langchain.chains.summarize import load_summarize_chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter, PythonCodeTextSplitter

openai_api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

#function to format nums in trillions, billions, millions fetched from data
def format_nums(num):
    if abs(num) >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return str(num)


#lookup for stocks : can be fetched from DB as well
stocks = {
    "Apple - 'AAPL'": {"name": "APPLE INC", "symbol": "AAPL", "cik": "0000320193"},
    "Alphabet - 'GOOG'": {"name": "Alphabet Inc.", "symbol": "GOOG", "cik": "0001652044"},
    "Facebook - 'META'": {"name": "META PLATFORMS INC", "symbol": "META", "cik": "0001326801"},
    "Amazon - 'AMZN'": {"name": "AMAZON COM INC", "symbol": "AMZN", "cik": "0001018724"},
    "Netflix - 'NFLX'": {"name": "NETFLIX INC", "symbol": "NFLX", "cik": "0001065280"},
    "Microsoft - 'MSFT'": {"name": "MICROSOFT CORP", "symbol": "MSFT", "cik": "0000789019"},
    "Tesla - 'TSLA'": {"name": "TESLA INC", "symbol": "TSLA", "cik": "0001318605"},
    "Nike - 'NKE'": {"name": "NIKE Inc", "symbol": "NKE", "cik": "0000320187"},
}

#fetch data from latest 10k of the company selected
def get_recommendation(stock_cik, question):
    company = Company(stock_cik["name"], stock_cik["cik"])
    doc = company.get_10K()
    text = TXTML.parse_full_10K(doc)

    llm = OpenAI(temperature=0.15, openai_api_key=openai_api_key)

    lts = int(len(text) / 3)
    lte = int(lts * 2)

    text_splitter = PythonCodeTextSplitter(chunk_size=3000, chunk_overlap=300)
    docs = text_splitter.create_documents([text[lts:lte]])

    #initialize the embeddings engine from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    #embed docs and combine with raw text in pseudo db. will call OpenAI API
    docsearch = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    query = question
    analysis = qa.run(query)

    return analysis.translate(str.maketrans("", "", "_*"))

st.set_page_config(page_title="Stock Analyzer", layout="wide", initial_sidebar_state="collapsed")
col1, col2 = st.columns((1, 3))
col1.header("< AB />")
col1.text("Powered by LangChain 🦜️🔗")
col1.write("This is an educational project only and should not be used for the purpose of stock trading. It is not intended to, and does not, constitute legal, financial, or professional advice of any kind. The user of this software assumes all responsibility for its use or misuse. ")
selected_stock = col1.selectbox("Select a stock", options=list(stocks.keys()), index=0)

#fetch stock info from yfinance
ticker = yf.Ticker(stocks[selected_stock]["symbol"])

#calc data range for last 30 days
end_date = datetime.now()
start_date = end_date - timedelta(days=360)

#fetch closing stock prices in last 30 days
data = ticker.history(start=start_date, end=end_date)
closing_prices = data["Close"]

#Visual of stock history
col1.line_chart(closing_prices, use_container_width=True)

#fetch company's description
long_desc = ticker.info["longBusinessSummary"]

#display desc in text box in second col
col2.title("Company Overview")
col2.write(long_desc)

#use yahooquery to get earnings and revenue
ticker_yq = Ticker(stocks[selected_stock]["symbol"])
earnings = ticker_yq.earnings

financials_data = earnings[stocks[selected_stock]["symbol"]]['financialsChart']['yearly']

df_financials = pd.DataFrame(financials_data)
df_financials['revenue'] = df_financials['revenue']
df_financials['earnings'] = df_financials['earnings']
df_financials = df_financials.rename(columns={"earnings": "yearly earnings", "revenue": "yearly revenue"})

numeric_cols = ['yearly earnings', 'yearly revenue']
df_financials[numeric_cols] = df_financials[numeric_cols].applymap(format_nums)
df_financials['date'] = df_financials['date'].astype(str)
df_financials.set_index('date', inplace=True)

#disp earnings and revenue in first col
col1.write(df_financials)

summary_det = ticker_yq.summary_detail[stocks[selected_stock]["symbol"]]

obj = yf.Ticker(stocks[selected_stock]["symbol"])

pe_ratio = '{0:.2f}'.format(summary_det["trailingPE"])
price_to_sales = summary_det["fiftyTwoWeekLow"]
target_price = summary_det["fiftyTwoWeekHigh"]
market_cap = summary_det["marketCap"]
ebitda = ticker.info["ebitda"]
tar = ticker.info["targetHighPrice"]
rec = ticker.info["recommendationKey"].upper()

#format large nums
market_cap = format_nums(market_cap)
ebitda = format_nums(ebitda)

#create lookup for additonal stock data
additional_data = {
    "P/E Ratio": pe_ratio,
    "52 Week Low": price_to_sales,
    "52 Week High": target_price,
    "Market Capitalisation": market_cap,
    "EBITDA": ebitda,
    "Price Target": tar,
    "Recommendation": rec
}

#disp additional data in col1
for key, value in additional_data.items():
    col1.write(f"{key}: {value}")

portfolio='[Amandeep Singh Bhalla](https://amandeep2230.github.io)'
st.subheader("Developed by {}.".format(portfolio))
#col2.title("Opportunities for traders")
print(f"**********\nstocks[selected_stock]\n**********\n{stocks[selected_stock]}\n\n**********\n")
#col2.write(get_recommendation(stocks[selected_stock], "What are this firm's key products and services?"))
#col2.write(get_recommendation(stocks[selected_stock], "What are the new products and growth opportunities for this firm? What are its unique strengths?"))
#col2.write(get_recommendation(stocks[selected_stock], "Who are this firms key competitors? What are the principal threats?"))
