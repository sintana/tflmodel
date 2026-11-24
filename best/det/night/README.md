# det · night — tf_det_v14 (night)

**ทำไมเป็นตัวนี้:** สูงสุดของ det กลางคืนบน s1 test ชุดเดียวกัน (unseen 285 รูปกลางคืน): **v14 0.614** · v12 0.597 · v13 0.571 · v10 0.546 · v11 0.524 · v6 0.509 · ดีกว่า v13 +0.043 แต่ CI แตะ 0 จึงนับว่าเสมอทางสถิติ

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `detect.tflite` | โมเดลที่ใช้บนกล้อง (int8/uint8 300×300) ใส่ใน eap เป็น `model/night_model.tflite` |
| `labelmap.txt` | ชื่อคลาส 10 คลาส เรียงตาม id 0–9 |
| `pipeline.config` | config การเทรน (TF1 Object Detection API) ตัวจริงที่ใช้เทรน |
| `eval/` | ผลทดสอบ: per-class AP บน test, ผลเทียบรุ่นก่อน (paired bootstrap) |
| `SHA256SUMS` | checksum ทุกไฟล์ |

## โมเดล

- **สถาปัตยกรรม:** SSDLite MobileDet 300×300 uint8 · fine-tune จาก tf_det_v10_night ckpt-40000 (สูตร tf_det_v12/v13) · geom center · เทรนบน RTX
- **input:** 300×300 uint8 · output `TFLite_Detection_PostProcess` · max detections 20
- **คลาส (id 0–9):** car, bike, truck, bus, taxi, pickup, trailer, ambulance, tuktuk, van
- **md5:** `4515830e5c722870a45b2cd269ed3fbf` · **sha256:** `7dcd6998f62453210945c6778380f69be90b995bc5e9ee896e2fdeb8260009a1` · ขนาด 3,576,976 byte

## การเทรน

- **เริ่มจาก:** `vca_runs/tf_det_v10_night/content/training/model.ckpt-40000`
- **steps:** 15000 · batch 32 · LR cosine base 0.005 · warmup 1000 step · QAT เริ่มที่ step 1000
- **augmentation:** random_horizontal_flip, ssd_random_crop, random_adjust_brightness, random_adjust_contrast, random_adjust_saturation, random_adjust_hue, random_rgb_to_gray
- **เครื่อง:** RTX 3090 batch 32 · fine-tune of tf_det_v10_night ckpt-40000 (tf_det_v12/v13 recipe, LR .005 + 1k warmup) · 15000 steps · LR cosine to 0 · QAT from step 1k · --geom center · best of [5000, 10000, 15000] on the s1 val (enixma) = 15000

## ข้อมูล

- **split:** s1 แบบ 80/10/10 (salt 147, 22 พ.ย.) · รูปเทรน 4,845 รูป → 14,887 ตัวอย่างใน TFRecord (รวม zoom crop) · val 537 ตัวอย่าง
- **geometry:** `center` (ครอปกลางภาพแบบเดียวกับ enixma) · clock-period True (ภาพที่นาฬิกา enixma ส่งมาให้โมเดลนี้ เวลา 06–18 = day)
- **กล่องต่อคลาส:** car 21,075 · bike 8,904 · pickup 8,363 · taxi 2,876 · van 2,547 · truck 2,417 · bus 2,073 · tuktuk 1,829 · trailer 1,548 · ambulance 583
- **แหล่งภาพ:** verified.lq 4,425 · eval.ruler_v2 153 · eval.t2test 185 · verified.trainset_v1 82

## ผล

- **val (s1, enixma geometry) ทุก step:** 5k 0.472 · 10k 0.490 · 15k 0.500 → เลือก step 15000

**เทียบ v13 · night · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_vs_v13_unseen.txt`)
```
tf_det_v13           0.571  [0.525, 0.633]
tf_det_v14           0.614  [0.565, 0.678]
tf_det_v14 - tf_det_v13: +0.043  [-0.005, +0.095]  P(better) 0.95  tie
```

**เทียบ v13 · night · all** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_vs_v13_all.txt`)
```
tf_det_v13           0.592  [0.551, 0.646]
tf_det_v14           0.621  [0.582, 0.676]
tf_det_v14 - tf_det_v13: +0.029  [-0.001, +0.067]  P(better) 0.97  tie
```

**รุ่นเก่าทุกตัวบน test เดียวกัน · night · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/compare_same_test_unseen.txt`)
```
tf_det_v13           0.571  [0.525, 0.633]
tf_det_v12           0.597  [0.542, 0.661]
tf_det_v11           0.524  [0.479, 0.590]
tf_det_v10           0.546  [0.497, 0.607]
tf_det_v6            0.509  [0.467, 0.573]
tf_det_v12 - tf_det_v13: +0.026  [-0.011, +0.074]  P(better) 0.88  tie
tf_det_v11 - tf_det_v13: -0.047  [-0.096, +0.009]  P(better) 0.05  tie
tf_det_v10 - tf_det_v13: -0.024  [-0.067, +0.019]  P(better) 0.11  tie
tf_det_v6 - tf_det_v13: -0.062  [-0.122, +0.000]  P(better) 0.03  tie
tf_det_v11 - tf_det_v12: -0.073  [-0.123, -0.021]  P(better) 0.00  WORSE
tf_det_v10 - tf_det_v12: -0.051  [-0.097, -0.015]  P(better) 0.00  WORSE
tf_det_v6 - tf_det_v12: -0.088  [-0.155, -0.017]  P(better) 0.01  WORSE
tf_det_v10 - tf_det_v11: +0.022  [-0.039, +0.078]  P(better) 0.73  tie
tf_det_v6 - tf_det_v11: -0.015  [-0.078, +0.059]  P(better) 0.34  tie
tf_det_v6 - tf_det_v10: -0.037  [-0.103, +0.037]  P(better) 0.15  tie
```

## ไฟล์อื่น

checkpoint, frozen graph, รายการรูปที่ใช้เทรน, code และเครื่องมือทำ eap ไม่ได้อยู่ใน repo นี้ ส่งแยกแบบ private
