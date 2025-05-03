import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="predict_type_personality.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Tipe Kepribadian MBTI")
st.write("Masukkan informasi berikut untuk memprediksi kepribadian kamu:")

# Form input pengguna
Age = st.slider("⏳Usia", 0, 50, 25)
IntroversionScore = st.slider("🧘Introversion Score", 0, 10, 5) # Changed: Using min, max, and default value
st.badge("Menunjukkan kecenderungan seseorang terhadap Ekstrovert (E) daripada Introvert(I).")
st.markdown("Skor tinggi → 😌lebih ekstrovert, suka interaksi sosial dan keramaian. ")
st.markdown("Skor rendah → 😶lebih pendiam, suka refleksi, butuh waktu sendiri untuk mengisi energi.")
SensingScore = st.slider("👀Sensing Score", 0, 10, 5)       # Changed: Using min, max, and default value
st.badge("Menunjukkan kecenderungan seseorang terhadap Sensing (S) daripada Intuition (N).")
st.markdown("Skor tinggi → 👁️suka informasi konkret, nyata, detail, dan pengalaman langsung.")
st.markdown("Skor rendah → 🌌cenderung intuitif, suka ide besar, pola, dan konsep.")
ThingkingScore = st.slider("🧠 Thinking Score", 0, 10, 5)     # Changed: Using min, max, and default value
st.badge("Menunjukkan kecenderungan terhadap Thinking (T) daripada Feeling (F).")
st.markdown("Skor tinggi → 🤖membuat keputusan berdasarkan logika dan objektivitas.")
st.markdown("Skor rendah → ❤️lebih mempertimbangkan perasaan orang lain dan empati saat mengambil keputusan.")
JudgingScore = st.slider("📅Judging Score", 0, 10, 5)
st.badge("Menunjukkan kecenderungan terhadap Judging (J) daripada Perceiving (P).")
st.markdown("Skor tinggi → 📋menyukai rencana, terorganisir, dan suka kepastian.")
st.markdown("Skor rendah → 🔄lebih fleksibel, spontan, dan terbuka terhadap perubahan.")

# --- Deskripsi MBTI & Tokoh Terkenal ---
mbti_info = {
    "INTJ": {
        "desc": "♟️ | 🧠 | 🧪 Logis, strategis, dan independen. Tipe pemimpin yang visioner dan berorientasi masa depan.",
        "tokoh": [" Elon Musk", "Mark Zuckerberg", "Stephen Hawking"],
        "pasangan": " INTJ analitis dan strategis bisa menemukan keseimbangan dengan pasangan yang fleksibel dan ideatif., Sehingga cocok dengan tipe kepribadian ENFP atau ENTP"
    },
    "INTP": {
        "desc": " 🔍 | 🤯 | 🧬 Analitis, penasaran, dan suka berpikir mendalam. Ahli teori dan konsep.",
        "tokoh": [" Albert Einstein", "Bill Gates", "Tina Fey"],
        "pasangan":" INTP penuh ide cocok dengan pasangan yang bisa membimbing secara struktur tapi tetap menghargai pemikiran mandiri. Sehingga cocok dengan tipe kepribadian ENTJ atau ENFJ"
    },
    "ENTJ": {
        "desc": " 🏆 | 🧭 | 🦁 Pemimpin alami, tegas, dan fokus pada efisiensi serta hasil.",
        "tokoh": [" Steve Jobs", "Margaret Thatcher", "Gordon Ramsay"],
        "pasangan":" ENTJ yang pemimpin alami cocok dengan pasangan yang mendukung dan berpikiran independen. Sehingga cocok dengan tipe kepribadian INFP atau INTP"
    },
    "ENTP": {
        "desc": " 💡 | 🗣️ | 🤹  Penuh ide, cepat berpikir, dan menyukai debat serta tantangan intelektual.",
        "tokoh": [" Thomas Edison", "Tom Hanks", "Robert Downey Jr."],
        "pasangan":" ENTP yang suka debat dan eksplorasi ide cocok dengan pasangan yang memberikan kedalaman dan arah. Sehingga cocok dengan tipe kepribadian INFJ atau INTJ"
    },
    "INFJ": {
        "desc": " 🔮 | 📖 | ✍️ Misterius, visioner, dan sangat empatik. Menggabungkan intuisi dengan idealisme.",
        "tokoh": [" Carl Jung", "Nicole Kidman", "Benedict Cumberbatch"],
        "pasangan":" INFJ idealis dan mendalam cocok dengan pasangan yang kreatif dan terbuka terhadap ide-ide besar. Sehingga cocok dengan tipe kepribadian ENFP atau ENTP "
    },
    "INFP": {
        "desc": " 🌈 | 🕊️ | ✨ Idealistis, penyayang, sangat kreatif dan introspektif. Peduli pada nilai dan makna hidup.",
        "tokoh": [" J.K. Rowling", "William Shakespeare", "Johnny Depp"],
        "pasangan":" INFP idealis dan berempati cocok dengan pasangan yang visioner dan berorientasi tujuan. Sehingga cocok dengan tipe kepribadian ENFJ atau ENTJ "
    },
    "ENFJ": {
        "desc": " 🌟 | 🤗 | 🗽 Karismatik, penyemangat, dan pemimpin alami yang peduli terhadap kesejahteraan orang lain.",
        "tokoh": [" Barack Obama", "Oprah Winfrey", "Jennifer Lawrence"],
        "pasangan":" ENFJ yang karismatik cocok dengan pasangan yang pendiam tapi memiliki visi kuat. Sehingga cocok dengan tipe kepribadian INFP atau INTP"
    },
    "ENFP": {
        "desc": " 📣 | ✨ | 😂 Antusias, kreatif, penuh energi. Suka eksplorasi dan membangun koneksi emosional.",
        "tokoh": [" Robin Williams", "Will Smith", "Ellen DeGeneres"],
        "pasangan":" ENFP yang kreatif dan ekspresif bisa berkembang bersama pasangan yang visioner dan mendalam. Sehingga cocok dengan tipe kepribadian INFJ atau INTJ"
    },
    "ISTJ": {
        "desc": " 🛡️ | 📊 | 🎖️ Konsisten, terstruktur, dan bertanggung jawab. Menghargai tradisi dan aturan.",
        "tokoh": [" Angela Merkel", "Natalie Portman", "George Washington"],
        "pasangan":" ISTJ terorganisir dan logis, cocok dengan pasangan yang spontan dan energik namun tetap menghargai struktur. Sehingga cocok dengan tipe kepribadian ESFP atau ESTP"
    },
    "ISFJ": {
        "desc": " 🧸 | ❤️ | 👼 Setia, tenang, perhatian. Menjaga harmoni dan suka menolong dengan cara praktis.",
        "tokoh": [" Beyoncé", "Mother Teresa", "Kate Middleton"],
        "pasangan": " ISFJ yang hangat dan penuh perhatian cocok dengan pasangan yang ekstrovert dan perhatian pada lingkungan. Sehingga cocok dengan tipe kepribadian ESFP atau ESTP"
    },
    "ESTJ": {
        "desc": " 📏 | 🧱 | 🧱 Tegas, terorganisir, dan suka mengatur. Sangat logis dan efisien.",
        "tokoh": [" Judge Judy", "Michelle Obama", "Frank Sinatra"],
        "pasangan":" ESTJ yang tegas dan berorientasi hasil cocok dengan pasangan yang lembut dan peduli. Sehingga cocok dengan tipe kepribadian ISFP atau INFP "
    },
    "ESFJ": {
        "desc": " 🤝 | 👩‍🍳 | 👨‍👩‍👧 Ramah, suka membantu, dan sangat peduli pada kebutuhan sosial orang lain.",
        "tokoh": [" Bill Clinton", "Jennifer Garner", "Taylor Swift"],
        "pasangan": " ESFJ yang suka membantu cocok dengan pasangan yang menghargai perhatian. Sehingga cocok dengan tipe kepribadian ISFP atau INFP"
    },
    "ISTP": {
        "desc": " 🛠️ | 😎 | 🏕️ Praktis, logis, dan suka menganalisis bagaimana sesuatu bekerja.",
        "tokoh": [" Clint Eastwood", "Michael Jordan", "Scarlett Johansson"],
        "pasangan": " ISTP mandiri dan logis cocok dengan pasangan yang ekspresif dan peduli secara sosial. Sehingga cocok dengan tipe kepribadian ESFJ atau ENFJ"
    },
    "ISFP": {
        "desc": " 🎨 | 🌿 | 🎶 Santai, sensitif, dan artistik. Suka kebebasan dan keindahan.",
        "tokoh": [" Britney Spears", "David Bowie", "Frida Kahlo"],
        "pasangan":" ISFP yang sensitif dan artistik bisa berkembang dengan pasangan yang suportif dan komunikatif. Sehingga cocok dengan tipe kepribadian ESFJ atau ENFJ"
    },
    "ESTP": {
        "desc": " 🏍️ | ⚡ | 🎯 Spontan, berani, suka aksi. Tipe petualang dan pengambil risiko.",
        "tokoh": [" Ernest Hemingway", "Eddie Murphy", "Bruce Willis"],
        "pasangan": " ESTP yang energik cocok dengan pasangan yang tenang dan suportif. Sehingga cocok dengan tipe kepribadian ISFJ atau ISTJ"
    },
    "ESFP": {
        "desc": " 🎉 | 💃 | 🎤 Ceria, ekspresif, dan menyukai perhatian. Tipe entertainer sejati.",
        "tokoh": [" Marilyn Monroe", "Adele", "Jamie Oliver"],
        "pasangan": " ESFP yang senang bersosialisasi cocok dengan pasangan yang memberi stabilitas. Sehingga cocok dengan tipe kepribadian ISFJ atau ISTJ"
    }
}

if st.button("🎯 Prediksi MBTI"):
    # Preprocessing input
    input_data = np.array([[Age, IntroversionScore, SensingScore, ThingkingScore, JudgingScore]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)
    predict_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"👏 MBTI mu adalah : **{predict_name.upper()}**")
    st.badge(f"**📝 Deskripsi :** {mbti_info.get(predict_name, {}).get('desc')}", color="blue")
    st.badge(f"**💃 Tokoh Terkenal :** {', '.join(mbti_info.get(predict_name, {}).get('tokoh', []))}", color='orange')
    st.markdown(f"<span style='color:purple'>**💝 Tipe Pasangan :** {mbti_info.get(predict_name, {}).get('pasangan', [0])}</span>", unsafe_allow_html=True)