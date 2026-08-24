import json
import pandas as pd

import os
from dotenv import load_dotenv
from openai import OpenAI


# 프로젝트 루트
products = pd.read_csv( "./data/pms_products.csv")
inventory = pd.read_csv("./data/pms_inventory.csv")
orders = pd.read_csv("./data/pms_orders.csv")


# --------------------------------------------------
# 실제 함수
# --------------------------------------------------
def get_price(product_name: str) -> str:
    """상품명(일부만 입력해도 됨)을 받아 판매가(원)를 반환한다. 가격/얼마 질문에 사용."""
    row = products[products["product_name"].str.contains(product_name, na=False)]
    if row.empty:
        return f"'{product_name}' 가격 정보를 찾지 못했습니다."
    r = row.iloc[0]
    return f"{r['product_name']} 판매가 {int(r['price']):,}원"

def get_stock(product_name: str) -> str:
    """상품명(일부만 입력해도 됨)을 받아 현재 재고 수량과 창고를 반환한다. 재고/품절 질문에 사용."""
    row = inventory[inventory["product_name"].str.contains(product_name, na=False)]
    if row.empty:
        return f"'{product_name}' 재고 정보를 찾지 못했습니다."
    r = row.iloc[0]
    return f"{r['product_name']} 재고 {int(r['stock'])}개 ({r['warehouse']})"

def get_order_status(order_id: str) -> str:
    """주문번호(예: O000106)를 받아 배송 상태를 반환한다. 주문/배송 추적에 사용."""
    row = orders[orders["order_id"] == order_id]
    if row.empty:
        return f"주문번호 {order_id}를 찾지 못했습니다."
    r = row.iloc[0]
    return f"주문 {order_id}: {r['product_name']} {int(r['quantity'])}개, 상태={r['status']}"

def search_product(keyword: str) -> str:
    """카테고리나 키워드로 상품을 검색해 이름 목록을 반환한다. '어떤 상품 있어?' 류에 사용."""
    hit = products[products["product_name"].str.contains(keyword, na=False) |
                   products["category"].str.contains(keyword, na=False)]
    if hit.empty:
        return f"'{keyword}' 관련 상품이 없습니다."
    return "검색 결과: " + ", ".join(hit["product_name"].head(5).tolist())



# --------------------------------------------------
# 함수 등록
# --------------------------------------------------
FUNCTIONS = {
    "get_price": get_price,
    "get_stock": get_stock,
    "get_order_status": get_order_status,
    "search_product": search_product,
}


# --------------------------------------------------
# Tool 스키마 자동 생성
# --------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": {
                    list(func.__code__.co_varnames[:func.__code__.co_argcount])[0]: {
                        "type": "string"
                    }
                },
                "required": [
                    list(func.__code__.co_varnames[:func.__code__.co_argcount])[0]
                ],
            },
        },
    }
    for name, func in FUNCTIONS.items()
]

# ============================
# OpenAI API를 사용하여 응답 생성
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
OPENAI_MODEL = "gpt-4.1-mini"


# --------------------------------------------------
# 질문 함수
# --------------------------------------------------
def ask(question):
    """질문을 받아 모델이 알맞은 도구를 골라 실행하고(자동), 최종 답변을 돌려준다."""

    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    messages.append(msg)

    for tc in msg.tool_calls:
        result = FUNCTIONS[tc.function.name](
            **json.loads(tc.function.arguments)
        )

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })

    final = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )

    return final.choices[0].message.content


# --------------------------------------------------
# 테스트
# --------------------------------------------------
for q in [
    "슬림핏 청바지 얼마야?",
    "이어버드 재고 있어?",
    "주문 O000106 배송 어디까지 왔어?",
    "패션의류 상품 보여줘",
]:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print("-" * 40)

