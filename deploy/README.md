# Agricultural Futures Intelligent Pricing System
# Streamlit Community Cloud Deployment

## Quick Deploy (3 steps)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Deploy: Agricultural Futures Pricing System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agri-futures-pricing.git
git push -u origin main
```

### 2. Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app" → Connect GitHub repo
3. Select: `agri-futures-pricing`
4. Main file path: `streamlit_app.py`
5. Click **Deploy**

### 3. Done!
Your app will be live at: `https://yourusername-agri-futures-pricing-app.streamlit.app`

---

## Project Structure for Cloud

```
deploy_streamlit_cloud/
├── streamlit_app.py          # Entry point (required by Streamlit Cloud)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme & server settings
├── src/                      # Core source code (copied from original)
│   ├── app.py                # Main application
│   ├── models/
│   ├── data/
│   ├── utils/
│   └── visualization/
└── data/                     # Sample data files
    ├── futures/              # Demo: A0_豆一.csv
    └── macro/                # Key economic indicators
```

## Features Available on Cloud

| Feature | Status | Notes |
|---------|--------|-------|
| Data Explorer | ✅ Full | 36 symbols (sample data for demo) |
| Causal Analysis | ✅ Full | PC algorithm + Bootstrap + Placebo |
| Pricing Model | ✅ Full | XGBoost + LightGBM + RF |
| Risk Assessment | ✅ Full | VaR/CVaR + Stress Test |
| Report Generation | ✅ Full | Auto-generate analysis reports |
| Weather Data | ⚠️ Limited | Uses embedded sample data |
| Remote Sensing | ⚠️ Limited | Uses embedded sample data |

## Customization

Edit `.streamlit/config.toml` to change theme colors, server settings.

Edit `requirements.txt` to add/remove packages.

## Troubleshooting

**"Module not found"**: Ensure all `src/` subdirectories have `__init__.py`

**"Data file missing"**: The system auto-generates demo data when real data is unavailable

**"Deployment failed"**: Check requirements.txt for version conflicts
