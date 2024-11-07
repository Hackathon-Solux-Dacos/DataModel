from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

df = pd.read_csv('book_data_embeddings.csv', encoding='utf-8-sig')

df['embedding'] = df['embedding'].apply(eval)

def find_similar_books(titles, top_n=1):
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
    similar_books = df_filtered.iloc[similar_indices][['Title', 'Author', 'Category', 'Cover_URL']].to_dict(orient='records')
    return similar_books[0]  # 가장 유사한 책 한 권만 반환

@app.route('/recommend', methods=['GET'])
def recommend():
    titles = request.args.getlist('title')
    if not titles:
        return jsonify({'error': 'No titles provided'}), 400
    
    try:
        similar_book = find_similar_books(titles)  
        return jsonify(similar_book) 
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 