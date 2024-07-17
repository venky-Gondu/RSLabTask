 # type: ignore
# convert tthe above code into streamlit code
import streamlit as st # type: ignore
import pandas as pd
st.title('Activity and Duration analysis')
st.sidebar.title('Activity and Duration analysis')
 # type: ignore
st.header('Upload data')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.write(data)
    # Repeat the same processing steps as above for the uploaded file
    data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str))
    data['duration'] = data['datetime'].diff().dt.total_seconds().fillna(0)

    inside_data = data[data['position'].str.lower() == 'inside']
    outside_data = data[data['position'].str.lower() == 'outside']

    inside_sum = inside_data.groupby('date')['duration'].sum().reset_index(name='inside_duration')
    outside_sum = outside_data.groupby('date')['duration'].sum().reset_index(name='outside_duration')

    #duration_summary = pd.merge(inside_duration, outside_duration, on='date', how='outer').fillna(0)

    activity_count = data.groupby(['date', 'activity']).size().unstack(fill_value=0).reset_index()

    #st.header("Duration Summary for Uploaded File")
    #st.dataframe(duration_summary)
    st.header("inside Duration")
    st.dataframe(inside_sum)
    st.header("outside Duration")
    st.dataframe(outside_sum)


    st.header("Activity Count for Uploaded File")
    st.dataframe(activity_count)