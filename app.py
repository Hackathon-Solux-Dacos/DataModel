from flask import Flask, request, Response
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

app = Flask(__name__)

df = pd.read_csv('book_data_embeddings.csv', encoding='utf-8-sig')

df['embedding'] = df['embedding'].apply(eval)

# 사용자 선호도 json을 dataframe으로 변환하는 함수
def get_preferences_from_spring(preference_json):
    try:
        # json -> df 으로 변환
        preferences_data = preference_json
        df = pd.DataFrame(preferences_data)

        # 출력확인
        #print("DataFrame 출력 결과:")
        #print(df)

        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

# 사용자 선호도와 가장 유사한 책을 반환하는 함수
def find_similar_books(preference_data, top_n=1):

    # 관심 있는 책 필터링
    interested_books = preference_data[preference_data['userReaction'] == 1]
    
    # 가중치 계산 (passTime을 가중치로 사용)
    weights = interested_books['passTime']
    
    # 관심 있는 책 제목에 해당하는 임베딩 가져오기
    embeddings = df[df['Title'].isin(interested_books['title'])]['embedding'].tolist()
    
    if not embeddings:
        raise ValueError("No valid titles found in the dataset.")
    
    # 가중치 기반 평균 임베딩 계산
    weighted_embeddings = np.average(embeddings, axis=0, weights=weights)
    
    # 입력된 책을 제외한 유사도 계산
    df_filtered = df[~df['Title'].isin(interested_books['title'])]
    similarities_filtered = cosine_similarity([weighted_embeddings], list(df_filtered['embedding']))
    
    # 유사도에 따라 책 정렬
    similar_indices = similarities_filtered.argsort()[0][-top_n:][::-1]
    
    # 유사한 책 정보 반환
    similar_books = df_filtered.iloc[similar_indices][['Title']].to_dict(orient='records')
    return similar_books[0]  # 가장 유사한 책 한 권만 반환

# 사용자 선호도를 추천하는 api
@app.route('/recommend', methods=['POST'])
def recommend():
    # request로 사용자 선호도 받기
    data = request.get_json()
    if not data:
        return Response(json.dumps({"error": "No data provided"}, ensure_ascii=False), status=400, mimetype='application/json')
    
    # 사용자 선호도 json -> df로 변환
    user_preferences_data = get_preferences_from_spring(data) 

    # 책추천 받기
    try:
        similar_book = find_similar_books(user_preferences_data)  
        return Response(json.dumps(similar_book, ensure_ascii=False), status=200, mimetype='application/json') 
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False), status=500, mimetype='application/json')
    
# 책 제목에 해당하는 책구절(content)반환
@app.route('/book/content', methods=['POST'])
def get_content_by_title():
    data = request.json  # 요청에서 title을 가져옵니다.
    title = data.get('title')

    if title:
        row = df[df['Title'] == title]
        if not row.empty:
            content = row.iloc[0]['Content']
            return Response(json.dumps(content, ensure_ascii=False), status=200, mimetype='application/json')
        else:
            return Response(json.dumps({"error": "해당 title에 대한 content가 없습니다."}, ensure_ascii=False), status=404, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "title 파라미터가 필요합니다."}, ensure_ascii=False), 400, mimetype='application/json')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True) 