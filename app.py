from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Flask 앱 생성
app = Flask(__name__)

# 임베딩이 포함된 데이터 로드
df = pd.read_csv('book_data_embeddings.csv', encoding='utf-8-sig')

# 임베딩을 리스트로 변환
df['embedding'] = df['embedding'].apply(eval)

# 코사인 유사도를 계산하여 가장 유사한 책을 찾는 함수
def find_similar_books(titles, top_n=1):
    # 선택된 책들의 임베딩을 평균화
    selected_embeddings = df[df['Title'].isin(titles)]['embedding'].tolist()
    if not selected_embeddings:
        raise ValueError("No valid titles found in the dataset.")
    
    # 평균 임베딩 계산
    average_embedding = np.mean(selected_embeddings, axis=0)
    
    # 입력된 책을 제외한 유사도 계산
    df_filtered = df[~df['Title'].isin(titles)]
    similarities_filtered = cosine_similarity([average_embedding], list(df_filtered['embedding']))
    
    # 유사도에 따라 책 정렬
    similar_indices = similarities_filtered.argsort()[0][-top_n:][::-1]
    
    # 유사한 책 정보 반환
    similar_books = df_filtered.iloc[similar_indices][['Title', 'Author', 'Category']].to_dict(orient='records')
    return similar_books[0]  # 가장 유사한 책 한 권만 반환

# API 엔드포인트 정의
@app.route('/recommend', methods=['GET'])
def recommend():
    titles = request.args.getlist('title')
    if not titles:
        return jsonify({'error': 'No titles provided'}), 400
    
    try:
        similar_book = find_similar_books(titles)  # 함수 호출 변경
        return jsonify(similar_book)  # 한 권의 책만 반환
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True) 