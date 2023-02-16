import streamlit as st
import  pandas as pd
import yfinance as yf
import datetime
from datetime import date
from PIL import Image
import prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

icon = Image.open("omega.png")
st.set_page_config(page_title='Foresight',page_icon=icon,layout='wide')

# Set Up
# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)


header = st.container()
with header:
    st.title('Foresight r-Ω')
    st.write('''##### Utilizing Prophet & Streamlit''')


project = st.container()
with project:
    st.write('### The App')
    st.text(
        '''
        This web application explores the stock trends and data
        on a number of top tech companies.
        ''')


about = st.container()
with about:
    st.write('### Instructions')
    st.text(
        '''
        How does this work?
        Here are the steps:
            1. Choose your stock of interest.
            2. Select how far into the future you want to predict.
            3. Observe the output.
        ''')


context = st.container()
with context:
    st.write('### Selection')
    #GOOGL (Alphabet Inc. Class A Common Stock)
    c1,c2=st.columns([1,1])
    option = c1.selectbox('Choose a Stock:', ('AAPL','GOOGL','MSFT','NVDA','TSLA'))
    increment = c2.number_input('Select years of prediction:',value=3,min_value=1,max_value=5,step=1)
    period = increment*365
    # Set timeframe
    today = datetime.date.today()
    start_date = today -datetime.timedelta(365*20)
    end_date = today    
    

visualize = st.container()
with visualize:
    # Get data
    @st.cache
    def stock_data():
        stock = yf.download(tickers=option,start=start_date,end=end_date, interval='1d')
        stock.reset_index(inplace=True)
        return stock
    # Data
    stock = stock_data()
    
    # Graph the data
    st.write('### Current Model')
    def plot_data():
        fig = go.Figure()
        fig.add_trace(go.Line(x=stock['Date'],y=stock['Close'],name='Closing Price'))
        fig.layout.update(title_text=f" {option} Closing Price",xaxis_rangeslider_visible=True)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price [$]')
        fig.update_layout(width=700,height=500,margin=dict(l=0,r=0,b=10,t=35))
        st.plotly_chart(fig)
    plot_data()

    # Tabular data
    st.write('###### Data from the Last Five Days')
    # st.table(stock.tail())
    t0 = stock.tail()
    t1 = go.Figure(data=[go.Table(
    header=dict(values=list(t0.columns),
        align='left',
        font=dict(size=20)),
    cells=dict(values=[t0.Date.dt.date,round(t0.Open,4),round(t0.High,4),round(t0.Low,4),round(t0.Close,4),round(t0['Adj Close'],4),t0.Volume],
        align='left',
        font=dict(size=18),
        height=26),
        )
    ])
    t1.update_layout(
        autosize=False,height=175,
        title_font_family='sans serif',
        margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(t1)


ml = st.container()
with ml:
    # Create the dataset
    train = stock[['Date','Close']]
    train = train.rename(columns={'Date':'ds','Close':'y'})

    # Create the model
    from prophet import Prophet
    model = Prophet()
    model.fit(train)
    future = model.make_future_dataframe(periods=period)
    prediction = model.predict(future)

    # plot the predictions
    st.write('### Predicted Model')
    future_fig = plot_plotly(model,prediction)
    future_fig.update_xaxes(title_text='Date')
    future_fig.update_yaxes(title_text='Price [$]')
    future_fig.update_layout(width=700,height=500)
    future_fig.update_layout(margin=dict(l=0,r=0,b=10,t=10))
    st.plotly_chart(future_fig)

    # fig2 = model.plot_components(prediction)
    # st.write(fig2)
    
    st.write('###### Data from the Last Five Days')
    tp0 = prediction.tail()
    tp1 = go.Figure(data=[go.Table(
    header=dict(values=list(tp0.columns[0:6]),
        align='left',
        font=dict(size=20)),
    cells=dict(values=[tp0.ds.dt.date,round(tp0.trend,4),round(tp0.yhat_lower,4),
        round(tp0.yhat_upper,4),round(tp0.trend_lower,4),round(tp0.trend_upper,4)],
        align='left',
        font=dict(size=18),
        height=26),
        )
    ])
    tp1.update_layout(height=175,margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(tp1)