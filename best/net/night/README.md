# net · night — tf_net_v14 (night)

**ทำไมเป็นตัวนี้:** สูงสุดของ net กลางคืนบน s1 test ชุดเดียวกัน (unseen 285 รูปกลางคืน): **v14 0.571** · v13 0.496 · v12 0.487 · v10 0.453 · ดีกว่า v13 +0.075 [+0.025, +0.122] อย่างมีนัยสำคัญ

## ไฟล์

| ไฟล์ | คืออะไร |
|---|---|
| `detect.tflite` | โมเดลที่ใช้บนกล้อง (int8/uint8 300×300) ใส่ใน eap เป็น `model/night_model.tflite` |
| `labelmap.txt` | ชื่อคลาส 10 คลาส เรียงตาม id 0–9 |
| `pipeline.config` | config การเทรน (TF1 Object Detection API) ตัวจริงที่ใช้เทรน |
| `eval/` | ผลทดสอบ: per-class AP บน test, ผลเทียบรุ่นก่อน (paired bootstrap) |
| `SHA256SUMS` | checksum ทุกไฟล์ |

## โมเดล

- **สถาปัตยกรรม:** SSD MobileNetV2 quantized 300×300 uint8 · เทรนจาก COCO (สูตร tf_net_v10) · geom center · เทรนบน RTX
- **input:** 300×300 uint8 · output `TFLite_Detection_PostProcess` · max detections 20
- **คลาส (id 0–9):** car, bike, truck, bus, taxi, pickup, trailer, ambulance, tuktuk, van
- **md5:** `94e80d6f54701fb7672c319204d101ad` · **sha256:** `8b15191e13a81bfa8ea1ef588278a96f3e2959e5418d949f61a8ef930ef32270` · ขนาด 4,943,096 byte

## การเทรน

- **เริ่มจาก:** `content/pre_trainded_model/ssd_mobilenet_v2_quantized_300x300_coco/model.ckpt`
- **steps:** 40000 · batch 32 · LR cosine base 0.004 · warmup 1000 step · QAT เริ่มที่ step 12000
- **augmentation:** random_horizontal_flip, ssd_random_crop, random_adjust_brightness, random_adjust_contrast, random_adjust_saturation, random_adjust_hue, random_rgb_to_gray
- **เครื่อง:** RTX 3090 batch 32 · from COCO (tf_net_v10 recipe, as tf_net_v13) · 40000 steps · LR cosine to 0 · QAT from step 12k · --geom center · best of [20000, 30000, 40000] on the s1 val (enixma) = 40000

## ข้อมูล

- **split:** s1 แบบ 80/10/10 (salt 147, 22 พ.ย.) · รูปเทรน 4,845 รูป → 14,887 ตัวอย่างใน TFRecord (รวม zoom crop) · val 537 ตัวอย่าง
- **geometry:** `center` (ครอปกลางภาพแบบเดียวกับ enixma) · clock-period True (ภาพที่นาฬิกา enixma ส่งมาให้โมเดลนี้ เวลา 06–18 = day)
- **กล่องต่อคลาส:** car 21,075 · bike 8,904 · pickup 8,363 · taxi 2,876 · van 2,547 · truck 2,417 · bus 2,073 · tuktuk 1,829 · trailer 1,548 · ambulance 583
- **แหล่งภาพ:** verified.lq 4,425 · eval.ruler_v2 153 · eval.t2test 185 · verified.trainset_v1 82

## ผล

- **val (s1, enixma geometry) ทุก step:** 20k 0.397 · 30k 0.414 · 40k 0.422 → เลือก step 40000

**เทียบ v13 · night · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_vs_v13_unseen.txt`)
```
tf_net_v13           0.496  [0.443, 0.554]
tf_net_v14           0.571  [0.522, 0.631]
tf_net_v14 - tf_net_v13: +0.075  [+0.025, +0.122]  P(better) 1.00  BETTER
```

**เทียบ v13 · night · all** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/gate_vs_v13_all.txt`)
```
tf_net_v13           0.499  [0.456, 0.548]
tf_net_v14           0.568  [0.527, 0.623]
tf_net_v14 - tf_net_v13: +0.070  [+0.031, +0.106]  P(better) 1.00  BETTER
```

**รุ่นเก่าทุกตัวบน test เดียวกัน · night · unseen** · mAP50 · paired bootstrap 1000× · CI 95% (`eval/compare_same_test_unseen.txt`)
```
tf_net_v13           0.496  [0.443, 0.554]
tf_net_v12           0.487  [0.436, 0.548]
tf_net_v10           0.453  [0.414, 0.510]
tf_net_v12 - tf_net_v13: -0.009  [-0.042, +0.028]  P(better) 0.35  tie
tf_net_v10 - tf_net_v13: -0.042  [-0.087, +0.014]  P(better) 0.08  tie
tf_net_v10 - tf_net_v12: -0.033  [-0.076, +0.009]  P(better) 0.07  tie
```

## ไฟล์อื่น

checkpoint, frozen graph, รายการรูปที่ใช้เทรน, code และเครื่องมือทำ eap ไม่ได้อยู่ใน repo นี้ ส่งแยกแบบ private
