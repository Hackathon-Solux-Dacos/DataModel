from flask import Flask, request, Response
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import requests

app = Flask(__name__)

# 데이터 로드 및 임베딩 변환
df = pd.read_csv('book_data_embeddings.csv', encoding='utf-8-sig')
df['embedding'] = df['embedding'].apply(eval)

def get_preferences_from_spring(preference_json):
    try:
        df = pd.DataFrame(preference_json)
        # 출력확인
        # print("DataFrame 출력 결과:", df)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def calculate_weights(interested_books):
    if len(interested_books) <= 1:
        return np.array([1.0])
    
    max_time = interested_books['passTime'].max()
    min_time = interested_books['passTime'].min()
    
    if max_time == min_time:
        weights = np.ones(len(interested_books))
    else:
        weights = (interested_books['passTime'] - min_time) / (max_time - min_time)
    
    return weights / weights.sum()

def find_similar_books(preference_data, top_n=1):
    # 관심 있는 책 필터링
    LIKE_REACTION = "좋아요"
    interested_books = preference_data[preference_data['userReaction'] == LIKE_REACTION]
    
    if interested_books.empty:
        return df.sample(n=1)['Title'].iloc[0]
    
    embeddings = df[df['Title'].isin(interested_books['title'])]['embedding'].tolist()
    if not embeddings:
        raise ValueError("No valid titles found in the dataset.")
    
    weights = calculate_weights(interested_books)
    weighted_embeddings = np.average(embeddings, axis=0, weights=weights)
    
    df_filtered = df[~df['Title'].isin(interested_books['title'])]
    similarities = cosine_similarity([weighted_embeddings], list(df_filtered['embedding']))
    similar_idx = similarities.argsort()[0][-top_n:][::-1]
    
    return df_filtered.iloc[similar_idx]['Title'].iloc[0]

@app.route('/recommend', methods=['POST'])
def recommend():
    # request로 사용자 선호도 받기
    data = request.get_json()
    if not data:
        return Response(json.dumps({"error": "No data provided"}, ensure_ascii=False), status=400, mimetype='application/json')
    
    # 사용자 선호도 json -> df로 변환
    user_preferences = get_preferences_from_spring(data) 

    # 책추천 받기
    try:
        recommendation = find_similar_books(user_preferences)
        return Response(json.dumps(recommendation, ensure_ascii=False), status=200, mimetype='application/json') 
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False), status=500, mimetype='application/json')

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

@app.route('/book/image', methods=['POST'])
def get_image_by_title():
    data = request.json  
    title = data.get('title')

    if title:
        row = df[df['Title'] == title]
        if not row.empty:
            image_url = row.iloc[0]['Cover_URL']
            print(image_url)

            try:
                response = requests.get(image_url)
                if response.status_code != 200:
                    return Response(json.dumps({"failed to retrieve image"}, ensure_ascii=False))
                return Response(response.content, content_type=response.headers['Content-Type'])
            except requests.exceptions.RequestException as e:
                return f"Error retrieving image: {str(e)}", 500

        else:
            return Response(json.dumps({"error": "해당 title에 대한 image_url이 없습니다."}, ensure_ascii=False), status=404, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "title 파라미터가 필요합니다."}, ensure_ascii=False), 400, mimetype='application/json')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True) 