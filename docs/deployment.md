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

## Public demo on Hugging Face Spaces (free)

Hugging Face Spaces runs Docker images for free (2 CPU cores, 16 GB memory), which is enough for PyMC and EconML. A typical free host with 512 MB of memory is not.

1. Create a free account at https://huggingface.co and make an access token with **write** permission (Settings, Access Tokens).
2. Create a Space: https://huggingface.co/new-space. Choose **Docker** (blank template), **Public**, free CPU hardware. Note its address, `https://huggingface.co/spaces/<your-username>/<space-name>`.
3. From the project folder:

```powershell
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git checkout -b space
copy deploy\hf_space_README.md README.md
git add README.md
git commit -m "Hugging Face Space configuration"
git push space space:main --force
git checkout main
```

When asked for a password, paste the access token. The Space builds the image (about 5 to 10 minutes the first time) and then shows the app at `https://<your-username>-<space-name>.hf.space`.

The `space` branch only differs from `main` in its `README.md` (Hugging Face reads its settings from the first lines of that file), so keep working on `main` and push to GitHub as usual.

### Things to know about the free demo

* The Space goes to sleep after a period of inactivity and takes a minute to wake.
* The first Bayesian analysis after a wake-up is slower because the sampler is compiled once (a few seconds).
* Uploaded files are used only to answer the request and are not kept afterwards.
