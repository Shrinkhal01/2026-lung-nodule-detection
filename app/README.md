# app/ (S: Phase 5)

**Gradio demo** that runs on the server and opens in your Mac's browser through an SSH tunnel.

- Pick a **test scan** → scroll through the slices with **detections drawn as circles**.
- A **slider** sets the operating point (false positives per scan).
- Click a detection → its **score**, **occlusion map** and **SHAP** bars.
- To open it from the Mac:
  ```bash
  ssh -L 7860:localhost:7860 <user>@<server>   # then open http://localhost:7860
  ```
