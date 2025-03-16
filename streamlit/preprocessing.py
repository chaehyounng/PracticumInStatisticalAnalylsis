# preprocessing.py
import re
import pandas as pd
from konlpy.tag import Mecab
import hanja

# 0. 사용자 정의 사전 (신조어, 고유명사 추가: ex) 캐즘, 아이오닉닉)
m = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# 1. 한자 및 특수문자 치환
def change_hanja_etc(x):
    x = re.sub('車', '차', x)
    x = re.sub('韓', '한국', x)
    x = re.sub('美', '미국', x)
    x = re.sub('日', '일본', x)
    x = re.sub('中', '중국', x)
    x = re.sub('英', '영국', x)
    x = re.sub('獨', '독일', x)
    x = re.sub("伊", "이탈리아", x)
    x = re.sub("佛", "프랑스", x)
    x = re.sub("亞", "아시아", x)
    x = re.sub("印", "인도", x)
    x = re.sub("比", "북한", x)
    x = re.sub("新", "새로운 ", x)
    x = re.sub('年', '매년', x)
    x = re.sub("前", '이전', x)
    x = re.sub("反", "반대", x)
    x = re.sub("强", "강자", x)
    x = re.sub("道", "도로", x)
    x = re.sub("業", "업적", x)
    x = re.sub("賞", "상", x)
    x = re.sub("弗", "달러", x)
    x = re.sub("對", "대결", x)
    x = re.sub('株', '주식 ', x)
    x = re.sub('州', '주', x)  # 외국 지역 단위 주
    x = re.sub('市', '시', x)
    x = re.sub('現', '현재', x)
    x = re.sub('社', '회사', x)
    x = re.sub('↑', '증가', x)
    x = re.sub('↓', '감소', x)
    x = hanja.translate(x, 'substitution')
    return x

# 2. 동의어 변환
def synonym(x):
    x = re.sub("전기 차", "전기차", x)
    x = re.sub(r"\b차\b", "자동차", x)
    x = re.sub("톱", "최고", x)
    x = re.sub("일렉트릭", "전기", x)
    x = re.sub("인니", "인도네시아", x)
    x = re.sub(r"\b말레이\b", "말레이시아", x)

    x = re.sub(r"\bSK\b", "SK", x)  # SK는 그대로 유지
    x = re.sub(r"(?<!S)\bK\b", "한국", x)  # 'K' 단독으로 존재 시
    x = re.sub(r"(?<!S)\bK(?=\s|\b)", "한국", x)  # 공백으로 이어지는 'K'
    x = re.sub(r"(?<!S)\bK(?=\w)", "한국", x)  # 단어와 붙어 있는 'K'

    x = re.sub("대한민국", "한국", x)
    x = re.sub("소나타", "쏘나타", x)
    x = re.sub("도요타", "토요타", x)
    x = re.sub("인니", "인도네시아", x)
    x = re.sub(r"\b말레이\b", "말레이시아", x)
    x = re.sub("어워즈", "상", x)
    x = re.sub("삼성 전자", "삼성전자", x)
    x = re.sub("인공 지능", "인공지능", x)
    x = re.sub('지난해', '전년', x)
    x = re.sub("리스크", "위험", x)
    x = re.sub('테크', '기술', x)
    x = re.sub("한해", "연간", x)
    x = re.sub("톱", "최고", x)
    x = re.sub("일렉트릭", "전기", x)
    x = re.sub("전기 차", "전기차", x)
    x = re.sub(r"\b차\b", "자동차", x)
    x = re.sub("톱", "최고", x)
    x = re.sub("일렉트릭", "전기", x)

    return x

# 3. 불용어 제거
delete_words = ["영상", "BIZ 플러스", "biz 플러스", "biz 플", "biz why", "biz FOCUS", "르포", "박영국의 디스", "업적데이트", "1보", "2보", "3보", "4보", "5보", "속보", "사진", "게시판", "주말 N", "QA", "그래픽", "신간",
                "종합", "위클리", "주간 화제의 뉴스", "카드뉴스", "팩트체크"]

delete_words.sort(key=len, reverse=True)

def word_delete(title):
    for word in delete_words:
      title = title.replace(word, "")

    return title.strip()

# 4. 텍스트 정제
def clean_text(title):
    # 한글, 영어(EV 때문에 남겨둠) 및 공백 제외한 문자 모두 제거, 중복생성된 공백 삭제
    title_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z\\s]", " ", title)
    title_clean = re.sub(" +", " ", title_clean)
    # 문자열 시작과 끝에 있는 공백 제거
    title_clean = title_clean.strip()
    return title_clean

# 5. 최종 전처리
def preprocessing_news(title):
    title = change_hanja_etc(title)
    title = synonym(title)
    title = word_delete(title)
    title = clean_text(title)
    return title

# 6. 토큰화
## 토큰 리스트트
def token_lst(text):
  allowed_pos = ['NNG', 'SL', 'NNP', 'VV', 'MAG']
  return [word for word, pos in m.pos(text) if pos in allowed_pos]

## 명사, 외국어, 동사만 남긴 token
def token(text):
  allowed_pos = ['NNG', 'SL', 'NNP', 'VV', 'MAG']
  words = [word for word, pos in m.pos(text) if pos in allowed_pos]
  return ' '.join(words)

# 7. 전처리 실행 함수
def preprocess_dataframe(df, column="제목"):
    df["clean_title"] = df[column].apply(preprocessing_news)
    df["token_lst"] = df["clean_title"].apply(token_lst)
    df["token"] = df["clean_title"].apply(token)
    return df
