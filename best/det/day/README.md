# det · day — tf_det_v13 (day)

**ทำไมเป็นตัวนี้:** สูงสุดของ det กลางวันบน s1 test ชุดเดียวกัน (unseen 561 รูปกลางวัน): **v13 0.679** · v12 0.661 · v10 0.614 · v11 0.599 · v6 0.587 · v14 0.671 (เทียบ v13 −0.008 [−0.030, +0.010] เสมอ จึงคง v13 · `eval/v14_vs_v13_unseen.txt`)

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `detect.tflite` | โมเดลที่ใช้บนกล้อง (int8/uint8 300×300) ใส่ใน eap เป็น `model/day_model.tflite` |
| `labelmap.txt` | ชื่อคลาส 10 คลาส เรียงตาม id 0–9 |
| `pipeline.config` | config การเทรน (TF1 Object Detection API) ตัวจริงที่ใช้เทรน |
| `eval/` | ผลทดสอบ: per-class AP บน test, ผลเทียบรุ่นก่อน (paired bootstrap) |
| `SHA256SUMS` | checksum ทุกไฟล์ |

## โมเดล

- **สถาปัตยกรรม:** SSDLite MobileDet 300×300 uint8 · fine-tune จาก tf_det_v10_day ckpt-40000 (สูตร tf_det_v12) · geom center · เทรนบน RTX
- **input:** 300×300 uint8 · output `TFLite_Detection_PostProcess` · max detections 20
- **คลาส (id 0–9):** car, bike, truck, bus, taxi, pickup, trailer, ambulance, tuktuk, van
- **md5:** `3fb61d83b9e11f8344afc678eabb2ed0` · **sha256:** `e614368da2ec420515fd71fab648bb71b873f585f78c35f534157ec203042900` · ขนาด 3,576,976 byte

## การเทรน

- **เริ่มจาก:** `vca_runs/tf_det_v10_day/content/training/model.ckpt-40000`
- **steps:** 15000 · batch 32 · LR cosine base 0.005 · warmup 1000 step · QAT เริ่มที่ step 1000
- **augmentation:** random_horizontal_flip, ssd_random_crop, random_adjust_brightness, random_adjust_contrast, random_adjust_saturation, random_adjust_hue, random_rgb_to_gray
- **เครื่อง:** RTX 3090 batch 32 · fine-tune of tf_det_v10_day ckpt-40000 (tf_det_v12 recipe, LR .005 + 1k warmup) · 15000 steps · LR cosine to 0 · QAT from step 1k · --geom center · best of [5000, 10000, 15000] on the s1 val (enixma) = 15000

## ข้อมูล

- **split:** s1 แบบ 75/10/15 ตอนเทรน (17 พ.ย.) · รูปเทรน 6,626 รูป → 20,454 ตัวอย่างใน TFRecord (รวม zoom crop) · val 884 ตัวอย่าง
- **geometry:** `center` (ครอปกลางภาพแบบเดียวกับ enixma) · clock-period True (ภาพที่นาฬิกา enixma ส่งมาให้โมเดลนี้ เวลา 06–18 = day)
- **กล่องต่อคลาส:** car 26,829 · pickup 16,784 · bike 11,419 · truck 5,356 · van 4,747 · taxi 3,011 · trailer 2,825 · bus 2,123 · ambulance 1,371 · tuktuk 988
- **แหล่งภาพ:** verified.lq 5,975 · eval.ruler_v2 278 · eval.t2test 253 · verified.trainset_v1 120

## ผล

- **val (s1, enixma geometry) ทุก step:** 5k 0.524 · 10k 0.532 · 15k 0.540 → เลือก step 15000

**ทุกรุ่นบน s1 test เดียวกัน (80/10/10) · day · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/compare_same_test_unseen.txt`)
```
tf_det_v13           0.679  [0.645, 0.720]
tf_det_v12           0.661  [0.626, 0.699]
tf_det_v11           0.599  [0.565, 0.637]
tf_det_v10           0.614  [0.579, 0.655]
tf_det_v6            0.587  [0.551, 0.625]
tf_det_v12 - tf_det_v13: -0.019  [-0.042, +0.001]  P(better) 0.03  tie
tf_det_v11 - tf_det_v13: -0.080  [-0.115, -0.049]  P(better) 0.00  WORSE
tf_det_v10 - tf_det_v13: -0.065  [-0.087, -0.044]  P(better) 0.00  WORSE
tf_det_v6 - tf_det_v13: -0.092  [-0.125, -0.060]  P(better) 0.00  WORSE
tf_det_v11 - tf_det_v12: -0.061  [-0.094, -0.031]  P(better) 0.00  WORSE
tf_det_v10 - tf_det_v12: -0.046  [-0.068, -0.022]  P(better) 0.00  WORSE
tf_det_v6 - tf_det_v12: -0.074  [-0.105, -0.041]  P(better) 0.00  WORSE
tf_det_v10 - tf_det_v11: +0.015  [-0.014, +0.048]  P(better) 0.83  tie
tf_det_v6 - tf_det_v11: -0.012  [-0.042, +0.018]  P(better) 0.21  tie
tf_det_v6 - tf_det_v10: -0.027  [-0.060, +0.003]  P(better) 0.04  tie
```

**gate ตอนรับรุ่น v13 (18 พ.ย., test ชุดเก่า 75/10/15) · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_round_unseen.txt`)
```
tf_det_v10           0.634  [0.603, 0.667]
tf_det_v12           0.671  [0.641, 0.703]
tf_det_v13           0.688  [0.659, 0.719]
tf_det_v12 - tf_det_v10: +0.037  [+0.020, +0.055]  P(better) 1.00  BETTER
tf_det_v13 - tf_det_v10: +0.054  [+0.038, +0.072]  P(better) 1.00  BETTER
tf_det_v13 - tf_det_v12: +0.017  [-0.000, +0.034]  P(better) 0.97  tie
```

## ไฟล์อื่น

checkpoint, frozen graph, รายการรูปที่ใช้เทรน, code และเครื่องมือทำ eap ไม่ได้อยู่ใน repo นี้ ส่งแยกแบบ private
