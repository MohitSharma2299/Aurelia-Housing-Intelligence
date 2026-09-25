import numpy as np
import pickle as pkl
import streamlit as st

with open('lr_cal_prac.pkl','rb') as f:
    obj = pkl.load(f)
model = obj['model']
cols = obj['columns']

st.title('California App')
In = []
for i in cols:
    v=st.number_input(f'Enter {i} value')
    In.append(v)
if st.button('click'):
    out=model.predict([In])
    st.success(f'The Median Housse Value is: {out}')