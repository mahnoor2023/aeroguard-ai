# AeroGuard AI - Streamlit upgrade (kaise lagana hai)

Repo me ye files replace / add karo (best.pt, packages.txt, results/ wahi rehne do):

| File | Kaam |
|---|---|
| app.py | REPLACE - naya advanced dashboard |
| core.py | NEW - detection engine (tiling, zone, heatmap, exports) |
| requirements.txt | REPLACE - `lapx` add hua (tracking ke liye) |
| .streamlit/config.toml | NEW - upload limit 200 MB + theme |

Local run:  `pip install -r requirements.txt`  then  `streamlit run app.py`
Streamlit Cloud: GitHub par push karo, app khud redeploy ho jati hai.

## Video test
`python make_test_video.py drone_photo.jpg test_video.mp4` se kisi bhi drone photo se test video banao,
phir app ke "Video analysis" tab me upload karo.
