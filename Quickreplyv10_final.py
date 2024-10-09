from flask import Flask, request, jsonify
from linebot import LineBotApi, WebhookHandler
from linebot.models import (MessageEvent, TextMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction, 
                            TemplateSendMessage, CarouselTemplate, CarouselColumn, URITemplateAction)
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import numpy as np
import ollama
from datetime import datetime
import datetime
import time
import json


app = Flask(__name__)

# Configuration for Neo4j
URI = "neo4j://localhost"
AUTH = ("neo4j", "Password")
driver = GraphDatabase.driver(URI, auth=AUTH)  # Persistent Neo4j connection

# Load the Sentence Transformer model once
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Chrome options for Selenium
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chromedriver_autoinstaller.install()

def get_ollama_response(prompt, history_chat):
    # Combine prompt with chat history
    history = "\n".join(history_chat)  # รวมประวัติการสนทนา
    full_prompt = (
        f"{history}\n"
        f"User: {prompt}\n"
        f"Bot (สวมบทบาทเป็นผู้จัดการโลตัส): "
    )

    try:
        # เพิ่มคำแนะนำในข้อความ prompt ให้ชัดเจน
        response = ollama.chat(model='supachai/llama-3-typhoon-v1.5', messages=[
            {
                'role': 'user',
                'content': full_prompt + (
                    "โปรดตอบคำถามในฐานะผู้จัดการโลตัส โดยให้ข้อมูลอย่างชัดเจน "
                    "เป็นทางการ อ้างอิงจากแหล่งข้อมูลที่เชื่อถือได้ ห้ามเดาคำตอบ "
                    "และควรมีความกระชับ ไม่เกิน 20 คำ เป็นภาษาไทย."
                ),
            },
        ])
        
        # ตรวจสอบโครงสร้างข้อมูลที่ได้รับ
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "Error: Unexpected response format."
    except Exception as e:
        return f"Error: {str(e)}"

# Caching data
promotion_cache = {}
cache_expiry = 600  # Cache expiry time in seconds (10 minutes)
last_cache_time = 0

access_token = '2ZaJOdqTNZ4q/5blyknNw+OCO9C1pPOCwhzA48d4alTTrOnbs2jRm/ydAlcgB8yl40Tp/pdv02dot4Jz75sunJCeZDB40J4NIV+unPpucs5Qg71sepDxuadKOr6cseR+ZpjnF2kJe6Pu6/IEQAuO3AdB04t89/1O/w1cDnyilFU='
secret = '52e6edb5ee988b218243f832e00e470b'
line_bot_api = LineBotApi(access_token)
handler = WebhookHandler(secret)

def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result]

def scrape_data_with_cache(url):
    global last_cache_time, promotion_cache
    if url in promotion_cache and time.time() - last_cache_time < cache_expiry:
        return promotion_cache[url]
    
    # If cache expired or no cache exists, scrape the data
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "sc-jlsrNB"))
        )
        html = driver.page_source
    except:
        driver.quit()
        return []
    
    driver.quit()
    soup = BeautifulSoup(html, "html.parser")
    product_elements = soup.find_all("div", {"class": "sc-jlsrNB"})

    result = []
    for element in product_elements:
        name = element.find("a", class_="sc-eicpiI").text.strip() if element.find("a", class_="sc-eicpiI") else "N/A"
        discounted_price = element.find("p", class_="sc-cVAmsi").find("span").text.strip() if element.find("p", class_="sc-cVAmsi") else "-"
        original_price = element.find("p", class_="sc-ksHpcM").text.strip() if element.find("p", class_="sc-ksHpcM") else "-"
        result.append({
            'product_name': name,
            'original_price': original_price,
            'discounted_price': discounted_price
        })

    # Cache the result and update the cache time
    promotion_cache[url] = result
    last_cache_time = time.time()
    return result

def store_chat_history(user_id, question, answer):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query = '''
    MERGE (u:User {id: $user_id})
    CREATE (q:Question {text: $question, timestamp: $timestamp})-[:ASKED_BY]->(u)
    CREATE (a:Answer {text: $answer, timestamp: $timestamp})-[:ANSWERED]->(q)
    '''
    run_query(query, parameters={"user_id": user_id, "question": question, "answer": answer, "timestamp": timestamp})

def check_chat_history(user_id, question):
    query = '''
    MATCH (u:User {id: $user_id})-[:ASKED_BY]->(q:Question {text: $question})-[:ANSWERED]->(a:Answer)
    RETURN a.text AS answer
    LIMIT 1
    '''
    result = run_query(query, parameters={"user_id": user_id, "question": question})
    if result:
        return result[0]["answer"]  # ถ้ามีประวัติการถามคำถามนี้ ให้ส่งคำตอบที่เก็บไว้
    return None  # ถ้าไม่มีประวัติ ให้ส่งค่า None กลับไป



# ฟังก์ชันสร้าง Quick Reply Buttons
def create_quick_reply_buttons():
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="สินค้าดังจากยูนิลีเวอร์", text="สินค้าดังจากยูนิลีเวอร์")),
        QuickReplyButton(action=MessageAction(label="สินค้าขายดี", text="สินค้าขายดี")),
        QuickReplyButton(action=MessageAction(label="สินค้าลดแรงแห่งปี", text="สินค้าลดแรงแห่งปี")),
        QuickReplyButton(action=MessageAction(label="อิ่มคุ้ม อาหารเจ", text="อิ่มคุ้ม อาหารเจ")),
        QuickReplyButton(action=MessageAction(label="ยิ่งซื้อยิ่งถูก", text="ยิ่งซื้อยิ่งถูก")),
        QuickReplyButton(action=MessageAction(label="เนื้อสัตว์ และอาหารทะเล", text="เนื้อสัตว์ และอาหารทะเล")),
        QuickReplyButton(action=MessageAction(label="น้ำ", text="น้ำ"))
    ])


# ฟังก์ชันสร้าง Carousel Template สำหรับโปรโมชั่น
def create_promotion_carousel():
    return CarouselTemplate(columns=[
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/Static1_UTT-W40-Static.jpg",
            title="สินค้าดังจากยูนิลีเวอร์",
            text="ดูสินค้ายูนิลีเวอร์ลดราคาพิเศษ",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/promotion/30th-annivesary-unilever-3-9oct")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/Static2_8octGHS-DD-1010-W40-CoinX5-App-Static.jpg",
            title="สินค้าขายดี",
            text="สินค้ายอดนิยม ลดราคาแรง",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/promotion/best-seller-flash")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/static3new_GHS-30Y-Iconic-W40-App-Static.jpg",
            title="สินค้าลดแรงแห่งปี",
            text="สินค้าลดสูงสุดแห่งปี",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/promotion/30th-anniversary-top-items")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/static4new_GHS-MSV-W40-Static.jpg",
            title="ยิ่งซื้อยิ่งถูก",
            text="ซื้อเยอะลดเยอะ",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/category/weekly-promotion")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/static5_GHS-J-Fest-W39-Static.jpg",
            title="อิ่มคุ้ม อาหารเจ",
            text="อาหารเจราคาพิเศษ",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/promotion/j-festival-26sep-11oct")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/mobile/online-marketing-banner/online-marketing-banner-1/online-image/2024/wk40-24/static6_8octGHS-Meat-seafood-W40-Static.jpg",
            title="เนื้อสัตว์ และอาหารทะเล",
            text="เนื้อสัตว์และอาหารทะเลราคาพิเศษ",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/category/meat")]
        ),
        CarouselColumn(
            thumbnail_image_url="https://www.lotuss.com/content/dam/aem-cplotusonlinecommerce-project/th/th/category/category-marketing-banner/images/2024/wk40-24/Water-Homeshelf-W39-Category-Banner.jpg",
            title="น้ำ",
            text="น้ำดื่มและเครื่องดื่มราคาพิเศษ",
            actions=[URITemplateAction(label="ดูเพิ่มเติม", uri="https://www.lotuss.com/th/category/milk-and-beverages-1/milk-and-beverages-water")]
        )
    ])


# ฟังก์ชันสำหรับการตอบกลับโปรโมชั่น
def send_promotion_reply(tk, user_id, msg,  category_url=None):
    if category_url:
        # ดึงข้อมูลสินค้าจาก URL หมวดหมู่
        products = scrape_data_with_cache(category_url)
        
        # สร้างข้อความสำหรับแสดงข้อมูลสินค้า
        product_details = ""
        for product in products:
            product_details += f"📌 **สินค้า: {product['product_name']}** 📌\n" \
                               f"   💵 ราคาปกติ: {product['original_price']}\n" \
                               f"   🔥 ราคาพิเศษ: {product['discounted_price']}\n\n"
        
        # ตรวจสอบว่ามีสินค้าในหมวดหมู่นั้นหรือไม่
        if not products:
            product_details = "ไม่พบสินค้าหรือข้อมูลในขณะนี้"

        # Store chat history with category message and product details
        store_chat_history(user_id, msg, product_details)

        # ตอบกลับด้วยข้อมูลสินค้าที่ดึงมา และปุ่ม Quick Reply
        quick_reply_buttons = create_quick_reply_buttons()

        line_bot_api.reply_message(
            tk, 
            [
                TextSendMessage(text=product_details),
                TextSendMessage(text="เลือกหมวดหมู่เพิ่มเติมได้จากปุ่มด้านล่าง", quick_reply=quick_reply_buttons)
            ]
        )
    else:
        # กรณีไม่มี URL หมายถึงให้แสดงโปรโมชั่นแคโรเซล
        carousel_template = create_promotion_carousel()
        quick_reply_buttons = create_quick_reply_buttons()

        template_message = TemplateSendMessage(
            alt_text="โปรโมชั่นลดราคาที่น่าสนใจ",
            template=carousel_template
        )
        # บันทึกประวัติแชท
        store_chat_history(user_id, msg, "โปรโมชั่นลดราคาที่น่าสนใจ")
        
        line_bot_api.reply_message(
            tk, 
            [
                template_message,
                TextSendMessage(text="เลือกหมวดหมู่เพิ่มเติมได้จากปุ่มด้านล่าง", quick_reply=quick_reply_buttons)
            ]
        )
# ตรวจสอบว่าผู้ใช้เคยมีประวัติในระบบหรือไม่
def check_user_exist(user_id):
    query = '''
    MATCH (u:User {id: $user_id})
    RETURN u
    '''
    result = run_query(query, {"user_id": user_id})
    return len(result) > 0  # ถ้ามี user อยู่แล้ว จะ return True

# เพิ่มประวัติผู้ใช้ใหม่ในระบบ
def add_new_user(user_id):
    query = '''
    MERGE (u:User {id: $user_id})
    '''
    run_query(query, {"user_id": user_id})

# ฟังก์ชันต้อนรับผู้ใช้ใหม่
def greet_new_user(tk, user_id):
    welcome_message = [
        TextSendMessage(text="สวัสดีค่ะ! ยินดีต้อนรับสู่บริการของเรา 😊"),
        TextSendMessage(text="ฉันเป็นแชทบอทที่จะช่วยคุณค้นหาสินค้าลดราคาที่น่าสนใจ รวมถึงโปรโมชั่นต่างๆ หากต้องการทราบข้อมูลสินค้าหรือโปรโมชั่นเพียงพิมพ์ชื่อสินค้าหรือคำค้นหาได้เลยค่ะ!")
    ]
    line_bot_api.reply_message(tk, welcome_message)
    add_new_user(user_id)  # เพิ่มผู้ใช้ใหม่ลงในฐานข้อมูล

# ฟังก์ชันสำหรับทักทายและแนะนำตัวเมื่อผู้ใช้พิมพ์คำทักทาย
def send_greeting(tk):
    greeting_message = [
        TextSendMessage(text="สวัสดีค่ะ! 😊"),
        TextSendMessage(text="ฉันเป็นแชทบอทที่พร้อมช่วยคุณค้นหาสินค้าลดราคาและโปรโมชั่นต่างๆ คุณสามารถพิมพ์'โปรโมชั่น' เพื่อดูโปรโมชั่นทั้งหมด 'ชื่อสินค้า' หรือคำค้นหาที่ต้องการได้เลยค่ะ!"),
        TextSendMessage(text="หรือหากต้องการดูหมวดหมู่โปรโมชั่น สามารถเลือกได้จากปุ่มด้านล่าง")
    ]
    quick_reply_buttons = create_quick_reply_buttons()  # สร้าง Quick Reply Buttons
    greeting_message.append(TextSendMessage(text="เลือกหมวดหมู่ได้เลยค่ะ", quick_reply=quick_reply_buttons))

    line_bot_api.reply_message(tk, greeting_message)

def search_product(product_name):
    # Go through cached data to search for products
    results = []
    for url, products in promotion_cache.items():
        for product in products:
            if product_name in product['product_name']:
                results.append({
                    'product_name': product['product_name'],
                    'original_price': product['original_price'],
                    'discounted_price': product['discounted_price']
                })
    return results


@app.route("/linebot", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    json_data = json.loads(body)
    signature = request.headers['X-Line-Signature']
    handler.handle(body, signature)
    
    msg = json_data['events'][0]['message']['text']
    tk = json_data['events'][0]['replyToken']
    user_id = json_data['events'][0]['source']['userId']

    # ตรวจสอบคำทักทาย
    greetings = ["สวัสดี", "ฮาย", "hi", "hello", "hey", "เฮ้", "เฮลโหล"]
    if msg in greetings:
        send_greeting(tk)
        return 'OK', 200

    # ตรวจสอบว่าเป็นผู้ใช้ใหม่หรือไม่
    if not check_user_exist(user_id):
        # ถ้าเป็นผู้ใช้ใหม่ ให้ส่งข้อความต้อนรับ
        greet_new_user(tk, user_id)
        return 'OK', 200
    
    # ตรวจสอบข้อความของผู้ใช้เพื่อแสดงโปรโมชั่นแคโรเซล
    if msg in ["โปรโมชั่น", "โปรโมชั่นลดราคา"]:
        send_promotion_reply(tk,user_id, msg)
    
    # ตรวจสอบการเลือกหมวดหมู่และดึงรายละเอียดสินค้าตาม URL ที่กำหนด
    elif msg == "สินค้าดังจากยูนิลีเวอร์":
        category_url = "https://www.lotuss.com/th/promotion/30th-annivesary-unilever-3-9oct"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "สินค้าขายดี":
        category_url = "https://www.lotuss.com/th/promotion/best-seller-flash"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "สินค้าลดแรงแห่งปี":
        category_url = "https://www.lotuss.com/th/promotion/30th-anniversary-top-items"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "ยิ่งซื้อยิ่งถูก":
        category_url = "https://www.lotuss.com/th/category/weekly-promotion"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "อิ่มคุ้ม อาหารเจ":
        category_url = "https://www.lotuss.com/th/promotion/j-festival-26sep-11oct"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "เนื้อสัตว์ และอาหารทะเล":
        category_url = "https://www.lotuss.com/th/category/meat"
        send_promotion_reply(tk, user_id, msg, category_url)
    
    elif msg == "น้ำ":
        category_url = "https://www.lotuss.com/th/category/milk-and-beverages-1/milk-and-beverages-water"
        send_promotion_reply(tk, user_id, msg, category_url)
    # Check category selection for specific promotion URLs
    category_urls = {
        "สินค้าดังจากยูนิลีเวอร์": "https://www.lotuss.com/th/promotion/30th-annivesary-unilever-3-9oct",
        "สินค้าขายดี": "https://www.lotuss.com/th/promotion/best-seller-flash",
        "สินค้าลดแรงแห่งปี": "https://www.lotuss.com/th/promotion/30th-anniversary-top-items",
        "ยิ่งซื้อยิ่งถูก": "https://www.lotuss.com/th/category/weekly-promotion",
        "อิ่มคุ้ม อาหารเจ": "https://www.lotuss.com/th/promotion/j-festival-26sep-11oct",
        "เนื้อสัตว์ และอาหารทะเล": "https://www.lotuss.com/th/category/meat",
        "น้ำ": "https://www.lotuss.com/th/category/milk-and-beverages-1/milk-and-beverages-water"
    }
    if msg in category_urls:
        send_promotion_reply(tk, user_id, msg, category_urls[msg])
        return 'OK', 200

    # Check if the input is a general product search term
    search_results = search_product(msg)
    if search_results:
        # Prepare product details to send back
        product_details = ""
        for product in search_results:
            product_details += f"📌 **สินค้า: {product['product_name']}** 📌\n" \
                               f"   💵 ราคาปกติ: {product['original_price']}\n" \
                               f"   🔥 ราคาพิเศษ: {product['discounted_price']}\n\n"

        store_chat_history(user_id, msg, product_details)  # Store search history

        # Reply with the product details and quick reply buttons
        quick_reply_buttons = create_quick_reply_buttons()
        line_bot_api.reply_message(
            tk, 
            [
                TextSendMessage(text=product_details),
                TextSendMessage(text="เลือกหมวดหมู่เพิ่มเติมได้จากปุ่มด้านล่าง", quick_reply=quick_reply_buttons)
            ]
        )
    else:
        # ถ้าไม่มีคีย์เวิร์ดที่กำหนดไว้หรือไม่พบสินค้า ให้ใช้ Ollama ตอบกลับ
        history_chat = check_chat_history(user_id, msg) or []
        ollama_response = get_ollama_response(msg, history_chat)
        store_chat_history(user_id, msg, ollama_response)

        line_bot_api.reply_message(
            tk,
            TextSendMessage(text=ollama_response)
        )

    return 'OK', 200
    
# Closing driver when done
@app.teardown_appcontext
def close_driver(exception):
    driver.close()
    if exception:
        print(f"Error closing driver: {exception}")

# Start the Flask server
if __name__ == '__main__':
    app.run()
