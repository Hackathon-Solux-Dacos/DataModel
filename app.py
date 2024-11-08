from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests 

app = Flask(__name__)

df = pd.read_csv('book_data_embeddings.csv', encoding='utf-8-sig')

df['embedding'] = df['embedding'].apply(eval)

def get_preferences_from_spring(preference_json):
    try:
        # json -> df 으로 변환
        preferences_data = preference_json
        df = pd.DataFrame(preferences_data)

        # 출력확인
        print("DataFrame 출력 결과:")
        print(df)

        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def find_similar_books(preference_data, top_n=1):

    # TODO 선호도 데이터로 알고리즘 수정필요 
    df = preference_data

    # selected_embeddings = df[df['Title'].isin(titles)]['embedding'].tolist()
    # if not selected_embeddings:
    #     raise ValueError("No valid titles found in the dataset.")
    # # 평균 임베딩 계산
    # average_embedding = np.mean(selected_embeddings, axis=0)
    # # 입력된 책을 제외한 유사도 계산
    # df_filtered = df[~df['Title'].isin(titles)]
    # similarities_filtered = cosine_similarity([average_embedding], list(df_filtered['embedding']))
    
    # # 유사도에 따라 책 정렬
    # similar_indices = similarities_filtered.argsort()[0][-top_n:][::-1]
    
    # # 유사한 책 정보 반환
    # similar_books = df_filtered.iloc[similar_indices][['Title', 'Author', 'Category', 'Cover_URL']].to_dict(orient='records')
    # return similar_books[0]  # 가장 유사한 책 한 권만 반환

@app.route('/recommend', methods=['POST'])
def recommend():
    # request로 사용자 선호도 받기
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # 사용자 선호도 json -> df로 변환
    user_preferences_data = get_preferences_from_spring(data) 

    try:
        similar_book = find_similar_books(user_preferences_data)  
        return jsonify(similar_book) 
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 