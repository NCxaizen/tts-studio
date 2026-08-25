# TTS Studio

A small web app for generating speech (Azure **or** ElevenLabs), mixing an optional
background-music bed at an adjustable volume, and exporting **PCM / A-law / µ-law WAV**
(44.1 kHz, 16 kHz, or 8 kHz telephony).

## Run locally
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5001  (needs `ffmpeg` on PATH, or it uses the bundled `imageio-ffmpeg`).

---

## Deploy to Render (public URL, free plan)

The app is Dockerized so ffmpeg is guaranteed on the server.

### 1. Put this folder in a GitHub repo
```
cd "TTS_Studio"
git init
git add .
git commit -m "TTS Studio"
git branch -M main
git remote add origin https://github.com/<you>/tts-studio.git
git push -u origin main
```
> `assets/default_music.wav` (~22 MB) is committed on purpose — it's the built-in track.

### 2. Create the service on Render
- Go to https://dashboard.render.com → **New +** → **Web Service**.
- Connect the GitHub repo.
- Render auto-detects the **Dockerfile**. Settings:
  - Environment: **Docker**
  - Plan: **Free**
  - Health check path: `/health`
- Click **Create Web Service**. First build takes a few minutes.

You'll get a public URL like `https://tts-studio-xxxx.onrender.com` that **anyone** can open.

### (Optional) Move keys to Render env vars
The keys are baked into `app.py` as defaults (per your choice to keep it open). To rotate
them without editing code, set these in Render → **Environment**:
- `ELEVEN_API_KEY`
- `AZURE_TTS_KEY`
- `AZURE_TTS_REGION` (default `eastus`)

Then blank the in-code defaults.

---

## ⚠️ Cost / abuse warning
This deployment is **fully open** and uses **your paid** Azure + ElevenLabs keys.
Anyone with the URL can generate audio and spend your credits. If usage or cost becomes a
concern, options are: add a shared password, require each visitor to paste their own key,
or take the site down. Set spending limits in the Azure and ElevenLabs dashboards.

## Notes
- Free Render instances **sleep after ~15 min idle**; the first request then takes ~30–60 s to wake.
- Generated files live on ephemeral disk and disappear on redeploy/restart — that's fine, they're downloads.
- Each browser gets its own session id, so simultaneous users don't overwrite each other's files.
