from flask import Flask, request, jsonify
from db import Database
from ollama_client import ask_llama
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # اجازه دسترسی فرانت جدا

# ⚙️ تنظیمات دیتابیس
db = Database(
    db_type="mysql",   # یا "sqlite"
    host="localhost",
    user="root",
    password="",
    database="chatbotdb"
)

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    دریافت پیام کاربر و اسم جدول
    واکشی رکورد مرتبط و ارسال به LLaMA3
    """
    try:
        user_msg = request.json.get("message")
        table = request.json.get("table", "persons")  # پیش‌فرض جدول
        user_name = request.json.get("name")  # برای فیلتر رکورد

        # واکشی رکوردها و ستون‌ها
        records, columns = db.fetch_all(table)

        # اگر user_name مشخص شده، فقط رکورد مرتبط را انتخاب کن
        if user_name:
            filtered_records = [r for r in records if r.get("نام کامل") == user_name]
            if filtered_records:
                records = filtered_records

        # آماده‌سازی متن دینامیک برای LLaMA3
        context = f"این داده‌ها از جدول '{table}' هستند:\n"
        for rec in records:
            row_text = " | ".join(f"{col}: {rec.get(col, 'عدم وجود!')}" for col in columns)
            context += f"- {row_text}\n"

        # prompt دقیق به مدل
        prompt = f"""
شما یک دستیار هوشمند هستید. اطلاعات جدول '{table}' به صورت زیر است:
{context}

لطفاً به دقت به ستون‌ها نگاه کن و فقط پاسخ صحیح را بده.  
اگر کاربر پرسید، فقط مقدار ستون مرتبط را بده، نه شماره ردیف یا ستون‌های دیگر.
اگر مقدار ستون وجود ندارد، 'عدم وجود!' را بنویس.
کاربر می‌پرسد: "{user_msg}"
"""

        response = ask_llama(prompt)
        return jsonify({"reply": response})

    except Exception as e:
        return jsonify({"reply": f"خطا در پردازش پیام: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
