import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import nn

# 데이터 불러오기
df_user_item = pd.read_pickle('./input/user_item.pkl')

# 모델 정의 AutoRec
class AutoRec(nn.Module):
    def __init__(self, num_inputs, hidden_units):
        super(AutoRec, self).__init__()

        self.encoder = nn.Linear(num_inputs, hidden_units)
        self.decoder = nn.Linear(hidden_units, num_inputs)
        
    def forward(self, x):
        x = torch.sigmoid(self.encoder(x))
        x = self.decoder(x)
        return x
    
# 모델 불러오기
num_inputs = df_user_item.shape[1]  # 입력 차원의 수
hidden_units = 500  # hidden layer의 unit 수
device = torch.device('cpu')  # device 설정
model = AutoRec(num_inputs, hidden_units).to(device)
model.load_state_dict(torch.load('./input/autorec_best_model.pt', map_location=torch.device('cpu')))

model.eval() 

# 유저가 선택한 아이템에 대한 추천 생성 함수
def user_free_inference(items, df_user_item, model, top_k=10):
    # Create a new user vector
    user_vector = np.zeros(df_user_item.shape[1])
    item_indices = []

    # Set the chosen items to the maximum value
    for item in items:
        if item in df_user_item.columns:
            item_index = df_user_item.columns.get_loc(item)
            user_vector[item_index] = df_user_item.values.max()
            item_indices.append(item_index)
        else:
            raise ValueError(f"Item {item} not found in the data")

    # Convert to tensor and move to the correct device
    user_vector = torch.FloatTensor([user_vector]).to(device)

    # Generate recommendations
    with torch.no_grad():
        outputs = model(user_vector)
        predicted_ratings = outputs.cpu().numpy()[0]

    # Remove the chosen items from the predictions
    predicted_ratings[item_indices] = -np.inf

    top_k_item_indices = np.argsort(-predicted_ratings)[:top_k]
    recommended_items = df_user_item.columns[top_k_item_indices]
    recommended_scores = predicted_ratings[top_k_item_indices]

    # Convert item and score to dictionary
    item_score_dict = dict(zip(recommended_items.tolist(), recommended_scores.tolist()))
    return item_score_dict

st.title('넥슨인 점심 추천을 해봅시다')

# 사용자 입력 받기
unique_items = sorted(df_user_item.columns.tolist())

multiselect_str = '''
선호하는 식당을 여러개 선택하세요\n
(단, 선택된 식당은 추천에서 제외됩니다)
'''
user_input = st.multiselect(multiselect_str, unique_items)
st.write('결과는 {식당명 : 예상 선호도}로 예상 선호도가 큰 순으로 10개가 나열됩니다.')

if user_input:
    item_score_dict = user_free_inference(user_input, df_user_item, model)
    result_df = pd.DataFrame(item_score_dict.items(), columns=['식당','예상 선호도'])
    print(result_df)
    st.dataframe(result_df)

        