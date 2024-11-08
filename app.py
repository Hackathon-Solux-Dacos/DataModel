from flask import Flask, request, Response, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
#from flask_sqlalchemy import SQLAlchemy
#import jaydebeapi

app = Flask(__name__)

#db_path = "testDBBB.mv.db"  # H2 데이터베이스 파일 경로
#h2_jar_path = "h2-2.3.232.jar" 

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
    """
    사용자 선호도 기반으로 책을 추천하는 함수
    Args:
        preference_data: 사용자 선호도 데이터프레임
        top_n: 추천할 책의 수
    Returns:
        top_n=1인 경우: 단일 책 제목
        top_n>1인 경우: 책 제목 리스트
    """
    # 관심 있는 책 필터링
    LIKE_REACTION = "좋아요"
    interested_books = preference_data[preference_data['userReaction'] == LIKE_REACTION]
    
    if interested_books.empty:
        result = df.sample(n=top_n)['Title']
        return result.iloc[0] if top_n == 1 else result.tolist()
    
    embeddings = df[df['Title'].isin(interested_books['title'])]['embedding'].tolist()
    if not embeddings:
        raise ValueError("No valid titles found in the dataset.")
    
    weights = calculate_weights(interested_books)
    weighted_embeddings = np.average(embeddings, axis=0, weights=weights)
    
    df_filtered = df[~df['Title'].isin(interested_books['title'])]
    similarities = cosine_similarity([weighted_embeddings], list(df_filtered['embedding']))
    similar_idx = similarities.argsort()[0][-top_n:][::-1]
    
    result = df_filtered.iloc[similar_idx]['Title']
    return result.iloc[0] if top_n == 1 else result.tolist()

# 가입시 선호도 기반으로 책 한권 추천
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
        recommendation = find_similar_books(user_preferences, top_n=1)
        return Response(json.dumps(recommendation, ensure_ascii=False), status=200, mimetype='application/json') 
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False), status=500, mimetype='application/json')

# 마이페이지에서 책 여러권 추천
@app.route('/recommend/multiple', methods=['POST'])
def recommend_multiple():
    data = request.get_json()
    if not data:
        return Response(json.dumps({"error": "No data provided"}, ensure_ascii=False), status=400, mimetype='application/json')
    
    user_preferences = get_preferences_from_spring(data) 

    try:
        recommendations = find_similar_books(user_preferences, top_n=30)
        return Response(json.dumps(recommendations, ensure_ascii=False), status=200, mimetype='application/json') 
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
            return Response(
                json.dumps({"cover_url": image_url}, ensure_ascii=False),
                status=200,
                mimetype='application/json'
            )
        else:
            return Response(
                json.dumps({"error": "해당 title에 대한 cover_url이 없습니다."}, ensure_ascii=False),
                status=404,
                mimetype='application/json'
            )
    else:
        return Response(
            json.dumps({"error": "title 파라미터가 필요합니다."}, ensure_ascii=False),
            status=400,
            mimetype='application/json'
        )

########################################################################################

model_path = 'path/to/kobert_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
kobert_model = BertForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = kobert_model(**inputs)
    _, predicted_class = torch.max(outputs.logits, dim=1)
    sentiment = "positive" if predicted_class.item() == 1 else "negative"  # Adjust according to your label mapping
    return sentiment

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_route():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return Response(json.dumps({"error": "No text provided"}, ensure_ascii=False), status=400, mimetype='application/json')
    
    sentiment = predict_sentiment(text)
    result = {
        "predict result": sentiment
    }
    return jsonify(result), 200





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True) 