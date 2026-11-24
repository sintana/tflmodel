# net · day — tf_net_v13 (day)

**ทำไมเป็นตัวนี้:** สูงสุดของ net กลางวันบน s1 test ชุดเดียวกัน (unseen 561 รูปกลางวัน): **v13 0.590** · v12 0.558 · v10 0.540 · v13 ดีกว่า v12 อย่างมีนัยสำคัญ · v14 0.590 (เทียบ v13 −0.000 [−0.019, +0.021] เสมอ และ AP ของ taxi กลางวันลดจาก 0.563 เหลือ 0.510 จึงคง v13 · `eval/v14_vs_v13_unseen.txt`)

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `detect.tflite` | โมเดลที่ใช้บนกล้อง (int8/uint8 300×300) ใส่ใน eap เป็น `model/day_model.tflite` |
| `labelmap.txt` | ชื่อคลาส 10 คลาส เรียงตาม id 0–9 |
| `pipeline.config` | config การเทรน (TF1 Object Detection API) ตัวจริงที่ใช้เทรน |
| `eval/` | ผลทดสอบ: per-class AP บน test, ผลเทียบรุ่นก่อน (paired bootstrap) |
| `SHA256SUMS` | checksum ทุกไฟล์ |

## โมเดล

- **สถาปัตยกรรม:** SSD MobileNetV2 quantized 300×300 uint8 · เทรนจาก COCO (สูตร tf_net_v10) · geom center · เทรนบน RTX
- **input:** 300×300 uint8 · output `TFLite_Detection_PostProcess` · max detections 20
- **คลาส (id 0–9):** car, bike, truck, bus, taxi, pickup, trailer, ambulance, tuktuk, van
- **md5:** `4375c46a347b1eb790a65da9e0239dde` · **sha256:** `463ac596ff7735616fc1fa718cea5ae3787547aae5dc6f144f6db08f177b75fc` · ขนาด 4,943,096 byte

## การเทรน

- **เริ่มจาก:** `content/pre_trainded_model/ssd_mobilenet_v2_quantized_300x300_coco/model.ckpt`
- **steps:** 40000 · batch 32 · LR cosine base 0.004 · warmup 1000 step · QAT เริ่มที่ step 12000
- **augmentation:** random_horizontal_flip, ssd_random_crop, random_adjust_brightness, random_adjust_contrast, random_adjust_saturation, random_adjust_hue, random_rgb_to_gray
- **เครื่อง:** RTX 3090 batch 32 · from COCO (tf_net_v10 recipe) · 40000 steps · LR cosine to 0 · QAT from step 12k · --geom center · best of [20000, 30000, 40000] on the s1 val (enixma) = 40000

## ข้อมูล

- **split:** s1 แบบ 75/10/15 ตอนเทรน (17 พ.ย.) · รูปเทรน 6,626 รูป → 20,454 ตัวอย่างใน TFRecord (รวม zoom crop) · val 884 ตัวอย่าง
- **geometry:** `center` (ครอปกลางภาพแบบเดียวกับ enixma) · clock-period True (ภาพที่นาฬิกา enixma ส่งมาให้โมเดลนี้ เวลา 06–18 = day)
- **กล่องต่อคลาส:** car 26,829 · pickup 16,784 · bike 11,419 · truck 5,356 · van 4,747 · taxi 3,011 · trailer 2,825 · bus 2,123 · ambulance 1,371 · tuktuk 988
- **แหล่งภาพ:** verified.lq 5,975 · eval.ruler_v2 278 · eval.t2test 253 · verified.trainset_v1 120

## ผล

- **val (s1, enixma geometry) ทุก step:** 20k 0.437 · 30k 0.458 · 40k 0.460 → เลือก step 40000

**ทุกรุ่นบน s1 test เดียวกัน (80/10/10) · day · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/compare_same_test_unseen.txt`)
```
tf_net_v13           0.590  [0.554, 0.631]
tf_net_v12           0.558  [0.522, 0.595]
tf_net_v10           0.540  [0.505, 0.578]
tf_net_v12 - tf_net_v13: -0.032  [-0.059, -0.007]  P(better) 0.01  WORSE
tf_net_v10 - tf_net_v13: -0.050  [-0.076, -0.023]  P(better) 0.00  WORSE
tf_net_v10 - tf_net_v12: -0.017  [-0.038, +0.003]  P(better) 0.04  tie
```

**gate ตอนรับรุ่น v13 (18 พ.ย., test ชุดเก่า 75/10/15) · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_round_unseen.txt`)
```
tf_net_v10           0.568  [0.535, 0.602]
tf_net_v12           0.577  [0.546, 0.609]
tf_net_v13           0.601  [0.570, 0.633]
tf_net_v12 - tf_net_v10: +0.009  [-0.007, +0.027]  P(better) 0.86  tie
tf_net_v13 - tf_net_v10: +0.033  [+0.010, +0.055]  P(better) 1.00  BETTER
tf_net_v13 - tf_net_v12: +0.024  [+0.002, +0.044]  P(better) 0.98  BETTER
```

## ไฟล์อื่น

checkpoint, frozen graph, รายการรูปที่ใช้เทรน, code และเครื่องมือทำ eap ไม่ได้อยู่ใน repo นี้ ส่งแยกแบบ private
