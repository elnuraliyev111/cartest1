#1. git clone
#2. python -m venv ./venv2
#3. .\venv2\Scripts\activate
#4. pip install requirements.txt
#5. streamlit run app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fitz  # PyMuPDF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_data_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    
    data = []
    lines = text.splitlines()
    for line in lines:
        if ' - ' in line:
            parts = line.split(' - ')
            if len(parts) == 2:
                plate_pattern = parts[0].strip()
                price_str = parts[1].strip()
                price_str = ''.join(filter(lambda x: x.isdigit() or x == ',', price_str))
                try:
                    # Handle cases where price might be in thousands (e.g., '7,000')
                    price = int(price_str.replace(',', ''))
                    data.append((plate_pattern, price))
                except ValueError:
                    print(f"Skipping line due to conversion error: {line}")

    return pd.DataFrame(data, columns=['plate_pattern', 'price'])

def round_up_to_next_10(x):
    return int(np.ceil(x / 10) * 10)

# Set page config for wider layout
st.set_page_config(layout="wide")

# Streamlit app
st.title('Price Prediction Based on Plate Length')

df = extract_data_from_pdf('Number97.pdf')

# Calculate the length of each plate pattern
df['length'] = df['plate_pattern'].apply(len)

option = st.radio(
    'Choose an option:',
    [
        'Special numbers',
        'Four-digit numbers',
        'Five-digit numbers',
        'Hexagonal numbers',
        'Special transport numbers (pick-up)',
        'Motorcycle numbers'
    ]
)

# Optionally: Add code to handle the selected option
if option == 'Special numbers':
    df_selected = df.head(110)
elif option == 'Four-digit numbers':
    df_selected = df.iloc[110:145]
elif option == 'Five-digit numbers':
    df_selected = df.iloc[145:271]
elif option == 'Hexagonal numbers':
    df_selected = df.iloc[271:477]
elif option == 'Special transport numbers (pick-up)':
    df_selected = df.iloc[477:536]
elif option == 'Motorcycle numbers':
    df_selected = df.iloc[536:]

X = df_selected[['length']]
y = df_selected['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
df_selected['predicted_price'] = model.predict(X)
df_selected['predicted_price'] = df_selected['predicted_price'].apply(round_up_to_next_10)

# Display the DataFrame

# Plotting
col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    # Add scatter plot for actual prices
    fig.add_trace(go.Scatter(
        x=df_selected['length'],
        y=df_selected['price'],
        mode='markers',
        name='Actual Prices',
        marker=dict(color='blue')
    ))

    # Add line plot for predicted prices
    fig.add_trace(go.Scatter(
        x=df_selected['length'],
        y=df_selected['predicted_price'],
        mode='lines',
        name='Fitted Line',
        line=dict(color='red')
    ))

    # User input for plate number
    default_plate_number = df_selected.iloc[0]['plate_pattern']
    plate_number = st.text_input('Enter Plate Number:', default_plate_number)

    input_length = len(plate_number)
    # Predict price for user input
    predicted_price = model.predict([[input_length]])[0]
    predicted_price_rounded = round_up_to_next_10(predicted_price)

    # Add marker for user input
    fig.add_trace(go.Scatter(
        x=[input_length],
        y=[predicted_price],
        mode='markers+text',
        name='User Input',
        marker=dict(color='green', size=12),
        text=[f'Predicted Price: ${predicted_price_rounded}'],
        textposition='top center'
    ))

    # Customize layout
    fig.update_layout(
        title='Using Machine Learning Models to Predict Car Plate Price',
        xaxis_title='Length of Plate Pattern',
        yaxis_title='Price ($)',
        xaxis=dict(type='linear'),
        yaxis=dict(type='linear')
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.write(f"### Plate Database for {option}:")
    st.write(df_selected)


st.markdown(f"""
    <div style="background-color: #f4f4f4; border-left: 5px solid #007bff; padding: 10px; margin-top: 20px;">
        <h3 style="color: #333;"> <strong>Forecasted Price for Plate Number '{plate_number}':</strong> <span style="color: #007bff;">${predicted_price_rounded}</span></h3>
    </div>
""", unsafe_allow_html=True)