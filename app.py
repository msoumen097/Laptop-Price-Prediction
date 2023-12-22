import streamlit as st
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

st.set_page_config(layout='wide',page_title='Laptop Price Prediction',page_icon='assets/favicon.ico')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Set page configuration
# st.set_page_config(layout="wide")
# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

col1,col2 = st.columns(2)
with col1:
    pass
with col2:
    github_profile_url = 'https://github.com/msoumen097'
    # Display GitHub link as a button
    st.markdown(f'<a href="{github_profile_url}" target="_blank" style="position: fixed; top: 60px; right: 10px; padding: 10px 10px; background-color: #24292e; color: #ffffff; text-decoration: none; border-radius: 30px;">''<img src="https://bitemycoin.com/wp-content/uploads/2018/06/GitHub-Logo.png" width="30" style="margin-right: 5px;"> GitHub''</a>',unsafe_allow_html=True)
# Containers
container1 = st.container()
container2 = st.container()
container3 = st.container()

# Content in the first container
with container1:
    col1, col2 = st.columns([8, 6])
    with col1:
        st.markdown("                      ")
        st.markdown("                      ")
        st.markdown("<p style='font-family: PT Serif; font-size: 40px;'>Laptop Price Prediction and Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: PT Serif; font-size: 15px;'>Welcome to Laptop Price Predictor! 🚀 Whether you're a tech enthusiast or a savvy shopper, our advanced machine learning model is here to make your laptop buying decision a breeze.</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("                      ")
        st.image("assets/MacBook Pro.png", width=300)

# Content in the second container
with container2:
    st.markdown("                      ")
    st.markdown("                      ")
    st.markdown("                      ")
    st.markdown("                      ")
    st.markdown("<p style='font-family: PT Serif; font-size: 30px; text-align: center;'>How It Works</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        cl1, cl2, cl3 = st.columns(3)
        with cl2:
            st.image('assets/c1.png', width=50)
        st.markdown("<p style='font-family: PT Serif;text-align : center;'><b>Customize Your Laptop</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: PT Serif;text-align : center'>Choose the specifications you want for your laptop, such as processor, RAM, storage, and more.</p>", unsafe_allow_html=True)
    with col2:
        cl1, cl2, cl3 = st.columns(3)
        with cl2:
            st.image('assets/c2.png', width=50)
        st.markdown("<p style='font-family: PT Serif;text-align : center'><b>Instant Price Prediction</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: PT Serif;text-align : center'>Our machine learning model analyzes your configuration and predicts the laptop's price accurately.</p>", unsafe_allow_html=True)
    with col3:
        cl1, cl2, cl3 = st.columns(3)
        with cl2:
            st.image('assets/c3.png', width=50)
        st.markdown("<p style='font-family: PT Serif;text-align : center'><b>Get Your Quote</b></p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family: PT Serif;text-align : center'>Receive the estimated price instantly and explore further options or adjustments.</p>", unsafe_allow_html=True)

with container3:
    col1, col2, col3 = st.columns(3)
    with col2:
        col1, col2 = st.columns([4, 2])  # Adjust the column widths as needed

# Content in the first column
        with col1:
            st.markdown('<style>h1</style>', unsafe_allow_html=True)
            st.markdown("<p style='font-family: PT Serif; font-size: 30px; text-align: center'>ML Model</p>", unsafe_allow_html=True)

# Content in the second column
        with col2:
            lottie_hello = load_lottieurl("https://lottie.host/51fb2057-1a91-4580-9e89-99f8866bee9c/rWlgX7gaiM.json")
            st_lottie(
            lottie_hello,
            speed=3,
            reverse=True,
            loop=True,
            quality="high",  # medium ; high
            height=100,
            width=150,
            key=None,
            )

col1, col2 = st.columns(2)
with col1:
    # brand
    # company = st.selectbox('Brand', df['Company'].unique(),placeholder='company_options',)
    company = st.selectbox('Brand', [''] + list(df['Company'].unique()))

    # type of laptop
    # type = st.selectbox('Type', df['TypeName'].unique())
    type = st.selectbox('Type', [''] + list(df['TypeName'].unique()))

    # Ram
    # ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    ram = st.selectbox('RAM(in GB)', ['',2, 4, 6, 8, 12, 16, 24, 32, 64])

    # weight
    weight = st.number_input('Weight of the Laptop')

    # Touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

    # IPS
    ips = st.selectbox('IPS', ['No', 'Yes'])

    # screen size
    screen_size = st.number_input('Screen Size')

with col2:
    # resolution
    resolution = st.selectbox('Screen Resolution', ['','1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

    # cpu
    cpu = st.selectbox('CPU', [''] + list(df['Cpu brand'].unique()))

    hdd = st.selectbox('HDD(in GB)', ['',0, 128, 256, 512, 1024, 2048])

    ssd = st.selectbox('SSD(in GB)', ['',0, 8, 128, 256, 512, 1024])

    gpu = st.selectbox('GPU', [''] + list(df['Gpu brand'].unique()))

    os = st.selectbox('OS', ['']+list(df['os'].unique()))

col1, col2, col3, col4, col5 = st.columns(5)

with col3:
    if st.button('Predict', type='primary'):
        try:
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0

            if ips == 'Yes':
                ips = 1
            else:
                ips = 0 
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
            query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 12)


            prediction = int(np.exp(pipe.predict(query)[0]))
            # st.title(f"The predicted price of this configuration is {prediction}")
            st.success(f"The predicted price is {prediction}")

        except Exception as e:
            # st.error(f"An error occurred: {str(e)}")
            st.error('Please Fill all details')

col1, col2, col3 = st.columns(3)
with col2:
    col1, col2 = st.columns([4, 2])  # Adjust the column widths as needed

# Content in the first column
    with col1:
        st.markdown('<style>h1</style>', unsafe_allow_html=True)
        st.markdown("<p style='font-family: PT Serif; font-size: 30px;  text-align: center'>Model Analysis</p>", unsafe_allow_html=True)

# Content in the second column
    with col2:
        lottie_hello = load_lottieurl("https://lottie.host/e94f37dd-9734-4e85-98db-5d3b8b25b27f/kp2XC1kvUr.json")
        st_lottie(
        lottie_hello,
        speed=3,
        reverse=True,
        loop=True,
        quality="low",  # medium ; high
        height=100,
        width=150,
        key=None,
        )       

col1, col2 = st.columns(2)
with col1:
    st.subheader("ML Model Chart")
    st.image('assets/model.png', use_column_width=True)

with col2:
    st.subheader("ML Prediction Chart")
    df_new = pd.read_csv('final_graph.csv')
    fig, ax = plt.subplots()
    ax.plot(df_new['index'], df_new['Price'], color='red', label='Actual Price')
    ax.plot(df_new['index'], df_new['predict'], color='blue', label='Predicted Price')
    ax.set_ylabel('Price')
    ax.set_title('Actual vs Predicted Prices')
    ax.legend()
    st.pyplot(fig)

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    # Read data from CSV
    df = pd.read_csv('data.csv')
    # Group by the 'Company' column and calculate the average 'Price' for each group
    data = df.groupby('Company')['Price'].mean().reset_index()
    # List of specified companies
    specified_companies = ['Acer', 'Apple', 'Asus', 'Dell', 'Google', 'HP', 'Razer', 'Samsung', 'Microsoft', 'Lenovo']
    # Update 'Company' column to 'Other' for companies not in the specified list
    data['Company'] = data['Company'].apply(lambda x: x if x in specified_companies else 'Other')
    # Create an interactive bar chart with hover information
    fig = px.bar(data, x='Company', y='Price', labels={'Price': 'Average Price'}, hover_data={'Price': ':.2f'})
    # Streamlit app
    st.title('Brand Wise')
    st.subheader('Average Laptop Costs')
    st.plotly_chart(fig)

col1, col2 = st.columns(2)
with col1:
    new_data = df.groupby('Ram').size().reset_index(name='Count')
    fig = px.bar(new_data, x='Ram', y='Count', labels={'Count': 'Laptop Count'}, title='Laptop Count by RAM(in GB)', hover_data={'Count': ':.0f'},)
    fig.update_xaxes(tickvals=new_data['Ram'])
    # Show the interactive plot
    st.title('Ram Impact')
    st.plotly_chart(fig)

with col2:
    data = {'Company': ['Acer', 'Apple', 'Asus', 'Other', 'Dell', 'Other', 'Google', 'HP', 'Other', 'Other', 'Lenovo', 'Other', 'Other', 'Microsoft', 'Razer', 'Samsung', 'Other', 'Other', 'Other'],
            'Count': [103, 21, 158, 3, 297, 3, 3, 274, 2, 3, 297, 54, 7, 6, 7, 9, 48, 4, 4]}
    new_df = pd.DataFrame(data)

    # Create an interactive pie chart using Plotly
    fig = px.pie(new_df, names='Company', values='Count',
                 title='Market share by Company',
                 hover_data={'Count': ':.2f'})
    st.title('Market Captured')
    fig.update_layout(showlegend=False)
    # Show the interactive plot
    st.plotly_chart(fig)

col1, col2, col3 = st.columns([5, 1, 4])

with col1:
    st.image('assets/extro.jpg')

with col3:
    col1, col2 = st.columns(2)
    st.markdown("<p style='font-family: PT Serif; font-size: 30px; text-align: left'>Project Devoloper</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: PT Serif; font-size: 30px;  text-align: left'>SOUMEN MONDAL</p>", unsafe_allow_html=True)
    st.markdown("I'm a passionate learner with a strong interest in data science. I'm currently focused on creating real-world projects using advanced machine learning techniques. I also have experience in data analysis, using Power BI, SQL, Streamlit, and Jupyter to build data-driven web applications.")

col1, col2, col3 = st.columns([5, 2, 5])
with col2:
    st.markdown("    ")
    st.markdown("    ")
    st.markdown("    ")
    st.markdown("    ")
    st.markdown("    ")
    st.markdown("    ")
    col1,col2,col3 = st.columns([7,1,7])
    with col2:
        icon_path = 'assets/favicon.ico'
        st.image(icon_path,width=20)
    st.markdown("<p style='font-family: PT Serif; font-size: 15px; text-align: center'>LAPTOP  PRICE</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: PT Serif; font-size: 15px; text-align: center'>© 2023. All rights reserved</p>", unsafe_allow_html=True)
