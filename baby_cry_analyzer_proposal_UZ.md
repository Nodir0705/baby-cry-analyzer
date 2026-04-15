# Chaqaloq Yig'lashini Tahlil Qiluvchi - Aqlli Alla Tizimi

## Loyiha Nomi

**"Chaqaloq yig'lashini tahlil qiluvchi va avtomatik tinchlantiruvchi aqlli tizim"**

---

## 1. Loyiha Tavsifi

Ushbu loyiha chaqaloq yig'lash ovozlarini real vaqt rejimida aniqlash va ularning sababini tasniflash uchun sun'iy intellektga asoslangan aqlli tizimni ishlab chiqishni maqsad qiladi. Tizim mikrofon orqali kelayotgan audio signallarni uzluksiz tinglaydi va chuqur o'rganish modellari yordamida chaqaloq yig'lashining turini aniqlaydi.

### Aniqlanadigan Holatlar

| Yig'lash Turi | Tavsifi |
|---------------|---------|
| **Ochlik** | Chaqaloq och, ovqatlantirish kerak |
| **Qo'rquv** | Chaqaloq qo'rqib ketgan |
| **Og'riq** | Chaqaloq og'riq his qilmoqda |
| **Noqulaylik** | Taglik almashtirish kerak, issiq/sovuq |

### Tizim Harakatlari

Tizim chaqaloq yig'layotganini aniqlasa:

1. **Telegram Bot** orqali ota-ona telefoniga xabar yuboradi
2. **Ehtimoliy sababni** ko'rsatadi (masalan: "Bola ochlik sababli yig'layapti")
3. **Avtomatik ravishda** tinchlantiruvchi musiqa yoki alla ijro etadi
4. **LED indikator** orqali vizual signal beradi

### Asosiy Maqsadlar

- Ota-onalarga chaqaloq ehtiyojini tezroq tushunishga yordam berish
- Ota-onalar stressini kamaytirish
- Chaqaloq parvarishini avtomatlashtirish
- **Onlayn va oflayn** rejimda ishlash

---

## 2. Tizim Talablari

| Talab | Qaror |
|-------|-------|
| **Qurilma** | Raspberry Pi 4 + USB Mikrofon + Karnay |
| **Bildirishnoma** | Telegram Bot |
| **Kechikish** | Real vaqt (<500ms) |
| **Ulanish** | Gibrid — oflayn ishlaydi, onlayn bo'lganda sinxronlaydi |

---

## 3. Tizim Arxitekturasi

### Qurilma Tuzilishi

```
┌─────────────────────────────────────────────────────┐
│               RASPBERRY PI QURILMASI                │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│  │   USB    │───▶│  Audio   │───▶│   AI Model   │  │
│  │ Mikrofon │    │  Bufer   │    │  (Inference) │  │
│  └──────────┘    └──────────┘    └──────────────┘  │
│                                         │          │
│                                         ▼          │
│                                  ┌──────────────┐  │
│                                  │  Yig'lash    │  │
│                                  │  Aniqlandi?  │  │
│                                  └──────────────┘  │
│                                    │          │    │
│                              HA ───┘          └─── YO'Q (davom)
│                                │                   │
│         ┌──────────────────────┼────────────┐     │
│         ▼                      ▼            ▼     │
│  ┌────────────┐    ┌────────────┐    ┌─────────┐ │
│  │  Telegram  │    │   Alla     │    │   LED   │ │
│  │   Xabar    │    │   Ijrosi   │    │  Holat  │ │
│  └────────────┘    └────────────┘    └─────────┘ │
│                                                   │
└─────────────────────────────────────────────────────┘
```

### Oflayn + Onlayn Ishlash

```
┌─────────────────────────────────────────────────────────────┐
│                    YIG'LASH ANIQLANDI                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Internet Bormi?           │
              └───────────────────────────────┘
                     │                 │
                    HA                YO'Q
                     │                 │
                     ▼                 ▼
        ┌─────────────────┐   ┌─────────────────────┐
        │ Telegram xabar  │   │ Lokal navbatga      │
        │ yuborish        │   │ saqlash (SQLite)    │
        └─────────────────┘   └─────────────────────┘
                     │                 │
                     └────────┬────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Alla ijro etish + │
                   │   LED indikator     │
                   └─────────────────────┘
                              │
                              ▼
              (Fon jarayoni internet tekshiradi)
                              │
                    Internet qaytganda:
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Navbatdagi xabarlarni│
                   │ Telegramga yuborish │
                   └─────────────────────┘
```

---

## 4. Qurilma Talablari

| Komponent | Misol | Taxminiy Narx |
|-----------|-------|---------------|
| Raspberry Pi 4 (4GB) | RPi 4 Model B | $55-60 |
| USB Mikrofon | ReSpeaker / har qanday USB mic | $10-30 |
| Karnay | 3.5mm yoki USB karnay | $10-20 |
| MicroSD Karta (32GB+) | SanDisk | $10 |
| Quvvat Manbai | Rasmiy RPi 5V/3A | $10 |
| (Ixtiyoriy) Korpus + LED | Demo uchun | $10 |

**Jami Taxminiy Narx: ~$100-130**

---

## 5. Dasturiy Ta'minot Arxitekturasi

```
project/
│
├── main.py                  # Asosiy fayl, hamma narsani boshqaradi
│
├── audio/
│   ├── capture.py           # Mikrofon kirishi, uzluksiz oqim
│   └── preprocessor.py      # MFCC ajratish, normalizatsiya
│
├── model/
│   ├── classifier.py        # Model yuklash va inference
│   └── cry_model.tflite     # O'qitilgan model fayli
│
├── actions/
│   ├── lullaby_player.py    # Audio fayllarni ijro etish
│   ├── led_controller.py    # GPIO LED boshqaruvi
│   └── notifier.py          # Telegram + oflayn navbat logikasi
│
├── storage/
│   └── event_store.py       # SQLite oflayn hodisalar uchun
│
├── config/
│   └── settings.py          # Chegaralar, Telegram token va h.k.
│
├── lullabies/               # Audio fayllar (.mp3/.wav)
│   ├── lullaby_01.mp3
│   └── lullaby_02.mp3
│
└── requirements.txt
```

---

## 6. Datasetlar

| Dataset | Tavsifi | Foydalanish |
|---------|---------|-------------|
| **Donate-a-Cry** | Turli yig'lash turlari (~450 namuna) | Asosiy o'qitish ma'lumotlari |
| **Baby Cry Dataset (Kaggle)** | Ochlik, og'riq, noqulaylik yorliqlari | Tasniflash o'qitishi |
| **ESC-50** | Atrof-muhit ovozlari | Salbiy namunalar (false positive kamaytirish) |
| **Shaxsiy yozuvlar** | O'zimiz yozib olingan namunalar | Lokal moslashish |

### Ma'lumotlarni Ko'paytirish Usullari

- Shovqin qo'shish
- Balandlik (pitch) o'zgartirish
- Tezlikni o'zgartirish
- Xona akustikasi simulyatsiyasi

---

## 7. Model Arxitekturasi

### Raspberry Pi uchun Real Vaqt Cheklovlari

| Omil | Tavsiya |
|------|---------|
| **Audio oynasi** | Har bir inference uchun 1-2 soniya audio |
| **Model hajmi** | <20MB (ideal <10MB) |
| **Model formati** | TensorFlow Lite yoki ONNX Runtime |
| **Xususiyatlar** | Mel-spektrogramma yoki MFCC |

### Tavsiya Etilgan Arxitektura

```
Audio (1.5s) → Mel-Spektrogramma → MobileNetV2 (kichik) → 4 sinf
                   │
            (64 mel diapazon × 94 vaqt freymi)
```

**RPi 4 da taxminiy inference vaqti: 100-200ms** ✅

### Model Jarayoni

1. **Xususiyat Ajratish**: Mel-spektrogramma (64 diapazon)
2. **Model**: MobileNetV2-small yoki maxsus CNN
3. **Chiqish**: 4 sinf (ochlik, qo'rquv, og'riq, noqulaylik)
4. **Ishonch chegarasi**: Harakat qilish uchun >0.7

---

## 8. Texnologiyalar Steki

| Kategoriya | Texnologiyalar |
|------------|----------------|
| **AI/ML** | Python, PyTorch/TensorFlow, Librosa |
| **Inference** | TensorFlow Lite, ONNX Runtime |
| **Qurilma** | Raspberry Pi 4, USB Mikrofon, Karnay |
| **Bildirishnoma** | Telegram Bot API |
| **Saqlash** | SQLite (oflayn hodisalar) |
| **Audio** | PyAudio, sounddevice |

---

## 9. Ishlab Chiqish Vaqt Jadvali (11 hafta)

| Bosqich | Davomiyligi | Vazifalar |
|---------|-------------|-----------|
| **1. Dataset va Preprocessing** | 2 hafta | Ma'lumotlarni yuklab olish, augmentation, xususiyat ajratish |
| **2. Model O'qitish** | 3 hafta | PC/Colab da o'qitish, optimallashtirish, TFLite ga o'girish |
| **3. RPi Asosiy Tizim** | 2 hafta | Audio yozib olish, inference loop, alla ijrosi |
| **4. Telegram + Oflayn Logika** | 1 hafta | Bot sozlash, SQLite navbat, sinxron logikasi |
| **5. Integratsiya va Sinov** | 2 hafta | Oxiridan-oxirigacha sinov, kechikishni optimallashtirish |
| **6. Hujjatlashtirish va Demo** | 1 hafta | Hisobot yozish, demo tayyorlash |

---

## 10. Baholash Mezonlari

| Mezon | Maqsad |
|-------|--------|
| **Aniqlik (Accuracy)** | >85% |
| **F1-Score** | >0.80 |
| **Inference Kechikishi** | <500ms |
| **Noto'g'ri Signal Darajasi** | Kuniga <5% |
| **Oflayn Funksionallik** | Asosiy funksiyalarning 100% |

---

## 11. Yakuniy Natijalar

- ✅ Chaqaloq yig'lashini tasniflash uchun o'qitilgan AI modeli
- ✅ Raspberry Pi asosidagi mustaqil qurilma
- ✅ Bildirishnomalar uchun Telegram Bot
- ✅ Avtomatik alla ijro etish tizimi
- ✅ Oflayn hodisalarni saqlash va sinxronlash
- ✅ Texnik hujjatlar / Diplom ishi
- ✅ Jonli namoyish

---

## 12. Kelajakdagi Yaxshilanishlar

- 📊 Tahlil paneli (veb-asosli)
- 👶 Har bir chaqaloq uchun shaxsiylashtirish
- 🏠 Aqlli uy integratsiyasi (Home Assistant)
- ⌚ Aqlli soat bildirishnomalari
- 🔊 Ovozli e'lonlar

---

## 13. Xavflarni Kamaytirish

| Xavf | Kamaytirish Usuli |
|------|-------------------|
| Kichik dataset | Kuchli augmentation + transfer learning |
| Noto'g'ri signallar (TV, boshqa bolalar) | Ishonch chegarasi + salbiy namunalar |
| RPi unumdorligi | Optimallashtirilgan TFLite model + kvantizatsiya |
| Tarmoq uzilishlari | Oflayn-birinchi dizayn navbat sinxronlash bilan |

---

*Hujjat Versiyasi: 1.0*  
*Oxirgi Yangilanish: 2025-yil Yanvar*
