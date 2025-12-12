Fintech Portfolio Starter — Python (FastAPI) + JS/HTML/CSS

https://dash.cloudflare.com/c800955a49e55b8a2f25aae0386536c2/pages/view/fintech-portfolio
https://console.cloud.google.com/run/detail/europe-west1/fintech-portfolio/observability/metrics?project=beaming-theorem-471715-v0
https://fintech-portfolio.pages.dev/


or for local : 

window.API_BASE = "";

# you need a Alpha Vantage API key for the backend to work
use a .env file with
ALPHA_VANTAGE_API_KEY=your_key_here
# To build and run the combined Docker image locally:

# Build the combined image
sudo docker build -f Dockerfile.full -t fintech-full .

# Run it
sudo docker run --rm -p 8000:8080 --env-file .env fintech-full


using the Dockerfile.full


sudo docker build -f Dockerfile.full -t fintech-full . && sudo docker run --rm -p 8000:8080 --env-file .env fintech-full
