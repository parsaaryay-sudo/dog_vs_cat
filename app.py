"""
تشخیص گربه یا سگ با استفاده از مدل Feature Extraction (Xception + Dense Head)
دقت تست این مدل در نوت‌بوک شما: 98.3٪ (بهترین مدل در بین سه مدل آموزش‌دیده)
"""

import streamlit as st
import numpy as np
from PIL import Image
import keras
import keras_hub

# -----------------------------------------------------------------------
# تنظیمات صفحه
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="تشخیص گربه یا سگ",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# استایل سفارشی (راست‌چین، فونت، کارت‌ها، نوار احتمال)
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Vazirmatn', sans-serif;
        direction: rtl;
    }

    .main {
        background: linear-gradient(160deg, #fdf6f0 0%, #eef3fb 100%);
    }

    .title-box {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        margin-bottom: 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #6C63FF 0%, #9d8cff 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(108, 99, 255, 0.25);
    }
    .title-box h1 {
        font-weight: 800;
        margin-bottom: 0.3rem;
        font-size: 2.1rem;
    }
    .title-box p {
        opacity: 0.9;
        font-size: 1rem;
        margin: 0;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        background: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        text-align: center;
    }
    .result-emoji {
        font-size: 3.4rem;
        line-height: 1;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0.4rem 0 0.1rem 0;
    }
    .result-sub {
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0.5rem 0;
    }
    .prob-label {
        width: 70px;
        font-weight: 600;
        text-align: right;
    }
    .prob-value {
        width: 55px;
        font-weight: 700;
        text-align: left;
        direction: ltr;
    }

    .stProgress > div > div > div > div {
        border-radius: 10px;
    }

    .uploadbox {
        border: 2px dashed #b8b8ff;
        border-radius: 16px;
        padding: 0.5rem;
        background: #f8f8ff;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# هدر
# -----------------------------------------------------------------------
st.markdown(
    """
    <div class="title-box">
        <h1>🐱 تشخیص گربه یا سگ 🐶</h1>
        <p>یک عکس آپلود کن تا مدل با دقت بالا (٪98.3 روی داده تست) بگه چند درصد گربه‌ست و چند درصد سگ</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------
# بارگذاری مدل‌ها (فقط یک بار، با کش استریم‌لیت)
# -----------------------------------------------------------------------
IMAGE_SIZE = (180, 180)
PRESET = "hf://keras/xception_41_imagenet"


@st.cache_resource(show_spinner="در حال بارگذاری مدل... (فقط بار اول کمی طول می‌کشد)")
def load_pipeline():
    # backbone آماده‌ی Xception برای استخراج ویژگی (feature extraction)
    conv_base = keras_hub.models.Backbone.from_preset(PRESET)
    conv_base.trainable = False

    # پیش‌پردازش تصویر مطابق چیزی که هنگام آموزش استفاده شده
    preprocessor = keras_hub.layers.ImageConverter.from_preset(
        PRESET, image_size=IMAGE_SIZE
    )

    # سر شبکه (Dense head) که خودتان روی ویژگی‌های Xception آموزش دادید
    head_model = keras.models.load_model("feature_extraction.keras")

    return conv_base, preprocessor, head_model


def predict(image: Image.Image, conv_base, preprocessor, head_model):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = keras.utils.img_to_array(image)
    batch = np.expand_dims(array, axis=0)

    preprocessed = preprocessor(batch)
    features = conv_base.predict(preprocessed, verbose=0)
    prob_dog = float(head_model.predict(features, verbose=0)[0][0])
    prob_cat = 1.0 - prob_dog
    return prob_cat, prob_dog


# -----------------------------------------------------------------------
# بدنه‌ی اصلی
# -----------------------------------------------------------------------
try:
    conv_base, preprocessor, head_model = load_pipeline()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(
        "بارگذاری مدل با خطا مواجه شد. مطمئن شوید فایل feature_extraction.keras "
        "کنار همین اسکریپت قرار دارد و پکیج‌های keras و keras_hub نصب هستند.\n\n"
        f"جزئیات خطا: {e}"
    )

st.markdown('<div class="uploadbox">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "عکس گربه یا سگ خود را اینجا رها کنید یا انتخاب کنید",
    type=["jpg", "jpeg", "png", "webp"],
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and model_ready:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="تصویر آپلود شده", use_container_width=True)

    with col2:
        with st.spinner("در حال تحلیل تصویر..."):
            prob_cat, prob_dog = predict(image, conv_base, preprocessor, head_model)

        if prob_dog >= 0.5:
            emoji, label = "🐶", "سگ"
            confidence = prob_dog
        else:
            emoji, label = "🐱", "گربه"
            confidence = prob_cat

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">این تصویر «{label}» است</div>
                <div class="result-sub">با احتمال {confidence*100:.1f}٪</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### جزئیات احتمال")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🐱 گربه — {prob_cat*100:.1f}٪**")
        st.progress(prob_cat)
    with c2:
        st.markdown(f"**🐶 سگ — {prob_dog*100:.1f}٪**")
        st.progress(prob_dog)

elif not model_ready:
    pass
else:
    st.info("منتظر آپلود عکس هستیم... 📷")

st.markdown(
    """
    <hr style="margin-top:2rem;">
    <p style="text-align:center; color:#999; font-size:0.85rem;">
        مدل مورد استفاده: Feature Extraction با بک‌بون Xception (دقت تست: ٪98.3)
    </p>
    """,
    unsafe_allow_html=True,
)
