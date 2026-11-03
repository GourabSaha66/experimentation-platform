# Deployment

The whole application (API and React app) is one Docker image, defined by the `Dockerfile` in the repository root. FastAPI serves the built React files, so there is only one service to run.

## Run it on your own computer

```powershell
# Windows PowerShell, from the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd frontend
npm install
npm run build
cd ..

uvicorn expplat.main:app --port 8000
# open http://127.0.0.1:8000
```

For front-end development, run `uvicorn expplat.main:app --port 8000` in one terminal and `npm run dev` inside `frontend/` in another, then open http://localhost:5173 (Vite forwards `/api` to the API).

With Docker: `docker build -t experiment-lab .` then `docker run -p 7860:7860 experiment-lab` and open http://localhost:7860.

## Public demo on Render (free plan, no card)

Render can run the Docker image on its free plan. The free plan has **512 MB of memory and a shared slice of one CPU**, and this app's peak memory with every method used is about 470 MB, so it fits but with little to spare. Treat the hosted demo as a convenience: if it is slow or restarts, the same app runs locally in a minute (see above), and the repository includes screenshots of every method.

(Hugging Face Docker Spaces would have more memory, but they now need a paid plan.)

1. Push the project to GitHub (done if you can see it on github.com).
2. Sign in at https://render.com with your GitHub account.
3. **New** then **Web Service**, choose **Build and deploy from a Git repository**, and select the `experimentation-platform` repository.
4. Settings:
   * **Language / Runtime:** Docker (Render finds the `Dockerfile` by itself)
   * **Branch:** `main`
   * **Instance type:** Free
5. Click **Create Web Service**. The first build takes about 10 to 15 minutes. The service address looks like `https://experimentation-platform-xxxx.onrender.com`.

### Things to know about the free demo

* The service falls asleep after about 15 minutes without visitors and takes roughly a minute to wake up. Open it once before showing it.
* The first Bayesian analysis after waking is the slowest (the sampler is compiled once). Later runs are faster.
* If the service shows **Out of memory** in its logs, only one heavy analysis should be run at a time; restarting the service from the Render dashboard clears it.
* Uploaded files are used only to answer the request and are not kept afterwards.