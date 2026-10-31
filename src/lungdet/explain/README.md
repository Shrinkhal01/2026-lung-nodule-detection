# explain/ (E: Phase 4)

**Photographs why each beep was flagged.**

- **Occlusion** → cover one small cube at a time and record the score drop → a heat map of **where the CNN looked**.
- **SHAP** → which features pushed the XGBoost score up or down.
- **Sanity check** → the same map from a **random-weights** network must look different, because in v2 layer4 Grad-CAM failed exactly this test.
- **Scored, not just shown:**
  - **hit** = the hottest point lies on the nodule
  - **concentration** = the share of heat that falls on the nodule
