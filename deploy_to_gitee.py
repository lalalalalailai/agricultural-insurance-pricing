# -*- coding: utf-8 -*-
"""
Gitee私有仓库部署 + Streamlit Cloud 发布
============================================
AI工具栈: Qwen3.6-Plus + GLM-5 + MiniMax M2.7 (100%国产)
"""
import os
import sys
import subprocess
import shutil
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = BASE_DIR
GITEE_REPO_NAME = "agri-futures-pricing"
PROJECT_NAME = "农险期货智能定价系统"
PROJECT_VERSION = "v1.0"

def log(msg, level="INFO"):
    tags = {"INFO": "[OK]", "WARN": "[!!]", "ERROR": "[ER]", "STEP": "[>>]"}
    print(f"{tags.get(level,'   ')} {level}: {msg}")

def check_prerequisites():
    log("Checking prerequisites...", "STEP")
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, text=True)
        log(f"Git: {r.stdout.strip()}")
    except:
        log("Git not found!", "ERROR")
        return False
    
    log(f"Python: {sys.version.split()[0]}")
    
    required = ["streamlit_app.py", "requirements.txt", ".streamlit/config.toml"]
    for f in required:
        fp = os.path.join(DEPLOY_DIR, f)
        if os.path.exists(fp):
            log(f"  OK {f} ({os.path.getsize(fp)/1024:.1f}KB)")
        else:
            log(f"  MISSING: {f}", "ERROR")
            return False
    
    py_count = 0
    for root, dirs, files in os.walk(os.path.join(DEPLOY_DIR, "src")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                py_count += 1
    log(f"Python files in src/: {py_count}")
    return True

def init_git_repo():
    log("Initializing Git repository...", "STEP")
    git_dir = os.path.join(DEPLOY_DIR, ".git")
    
    if os.path.exists(git_dir):
        subprocess.run(["git", "rm", "-r", "--cached", "."], cwd=DEPLOY_DIR,
                      capture_output=True)
        log("Re-initializing existing repo...")
    
    r = subprocess.run(["git", "init"], cwd=DEPLOY_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Git init failed: {r.stderr}", "ERROR")
        return False
    
    r = subprocess.run(["git", "config", "user.name"], cwd=DEPLOY_DIR,
                       capture_output=True, text=True)
    if not r.stdout.strip():
        log("Git user not configured. Setting defaults...")
        subprocess.run(["git", "config", "user.name", "Competition Team"], cwd=DEPLOY_DIR)
        subprocess.run(["git", "config", "user.email", "team@competition.edu.cn"], cwd=DEPLOY_DIR)
    
    log("Git repo initialized OK")
    return True

def add_and_commit():
    log("Adding files to staging...", "STEP")
    subprocess.run(["git", "add", "-A"], cwd=DEPLOY_DIR, capture_output=True)
    
    r = subprocess.run(["git", "status", "--short"], cwd=DEPLOY_DIR,
                       capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l]
    log(f"Files staged: {len(lines)}")
    
    msg = f"{PROJECT_NAME} {PROJECT_VERSION} - {datetime.now().strftime('%Y-%m-%d')}"
    log(f"Commit message: {msg}")
    
    r = subprocess.run(["git", "commit", "-m", msg], cwd=DEPLOY_DIR,
                      capture_output=True, text=True)
    if "nothing to commit" in (r.stdout or "") + (r.stderr or ""):
        log("No changes to commit (clean state)")
    elif r.returncode == 0:
        log("Committed successfully!")
        r2 = subprocess.run(["git", "log", "--oneline", "-3"], cwd=DEPLOY_DIR,
                            capture_output=True, text=True)
        log(f"Recent commits:\n{r2.stdout.strip()}")
    else:
        log(f"Commit warning: {r.stderr}", "WARN")
    return True

def show_private_guide():
    log("=" * 60, "STEP")
    log("PLAN A: Gitee Private Repo + VPS (Code 100% Private)", "STEP")
    log("=" * 60)
    
    print("""
+----------------------------------------------------------+
|  GITEE PRIVATE REPO SETUP STEPS                            |
+----------------------------------------------------------+
|  Step 1: Open browser -> https://gitee.com/projects/new   |
|  Step 2: Fill in:                                          |
|          Repository name: agri-futures-pricing             |
|          Visibility: *** PRIVATE *** (must select!)       |
|          DO NOT initialize with README                     |
|  Step 3: After creation, you get:                          |
|          https://gitee.com/YOUR_USERNAME/agri-futures-pricing.git |
|  Step 4: Run git push commands below                       |
+----------------------------------------------------------+
""")
    
    username = input("Enter your Gitee username: ").strip()
    if not username:
        username = "YOUR_USERNAME"
        log("Using placeholder username", "WARN")
    
    remote_url = f"https://gitee.com/{username}/{GITEE_REPO_NAME}.git"
    
    r = subprocess.run(["git", "remote", "add", "origin", remote_url],
                      cwd=DEPLOY_DIR, capture_output=True, text=True)
    if "already exists" in (r.stderr or ""):
        subprocess.run(["git", "remote", "set-url", "origin", remote_url],
                      cwd=DEPLOY_DIR, capture_output=True)
        log("Remote URL updated")
    else:
        log(f"Remote added: origin -> {remote_url}")
    
    print(f"""
+----------------------------------------------------------+
|  PUSH COMMANDS (run in terminal):                         |
|                                                          |
|  cd "{DEPLOY_DIR}"                                     |
|  git push -u origin master --force                         |
|                                                          |
|  NOTE: First push will ask for Gitee login credentials     |
+----------------------------------------------------------+
""")
    
    do_push = input("Push to Gitee now? (y/n): ").strip().lower()
    if do_push == "y":
        log("Pushing to Gitee private repo...")
        r = subprocess.run(["git", "push", "-u", "origin", "master", "--force"],
                          cwd=DEPLOY_DIR, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            log("Pushed to Gitee successfully!")
            log(f"Repo URL: https://gitee.com/{username}/{GITEE_REPO_NAME} [PRIVATE]")
        else:
            log(f"Push failed: {r.stderr}", "ERROR")
            log("Please run manually: git push -u origin master")
    else:
        log("Skipped auto-push. Push manually when ready.")
    
    print("""
+----------------------------------------------------------+
|  NEXT STEP: DEPLOY TO SERVER                               |
+----------------------------------------------------------+
|  Since you chose PRIVATE repo, Streamlit Cloud CANNOT be   |
|  used (it requires public GitHub repos only).              |
|                                                          |
|  Recommended VPS options (Chinese cloud providers):         |
|  [BEST] Alibaba Cloud Light Server (~50 RMB/month)         |
|    -> https://www.aliyun.com/product/swas                 |
|    -> Ubuntu 22.04, 2C2G RAM                              |
|                                                          |
|  [ALT] Tencent Cloud Light Server (~50 RMB/month)         |
|  [FREE] Use ngrok/frp to expose local port (temporary)     |
+----------------------------------------------------------+
|  Deploy commands on server:                                |
|    git clone https://gitee.com/{user}/agri-futures-pricing.git |
|    cd agri-futures-pricing                                 |
|    pip install -r requirements.txt                         |
|    streamlit run streamlit_app.py --server.port 8501       |
+----------------------------------------------------------+
""".format(user=username))
    
    return f"https://gitee.com/{username}/{GITEE_REPO_NAME}"

def show_public_guide():
    log("=" * 60, "STEP")
    log("PLAN B: GitHub Public Repo + Streamlit Cloud", "STEP")
    log("=" * 60)
    
    print("""
+----------------------------------------------------------+
|  WARNING: This plan makes your CODE PUBLIC!                |
|  If you want code privacy, use PLAN A instead.             |
+----------------------------------------------------------+
""")
    
    confirm = input('Type "PUBLIC" to confirm public deployment: ').strip()
    if confirm != "PUBLIC":
        log("Cancelled. Use PLAN A for private deployment.")
        return None
    
    username = input("Enter your GitHub username: ").strip()
    if not username:
        username = "YOUR_USERNAME"
    
    github_url = f"https://github.com/{username}/{GITEE_REPO_NAME}"
    
    print(f"""
+----------------------------------------------------------+
|  GITHUB PUBLIC REPO + STREAMLIT CLOUD                      |
+----------------------------------------------------------+
|  Step 1: Create GitHub public repo                         |
|    -> https://github.com/new                               |
|    -> Repository name: agri-futures-pricing               |
|    -> MUST select Public                                   |
|                                                          |
|  Step 2: Push code                                        |
|    git remote add github {github_url}.git                |
|    git push -u github main                                 |
|                                                          |
|  Step 3: Deploy on Streamlit Cloud                         |
|    -> https://share.streamlit.io                           |
|    -> Deploy an app -> From GitHub                         |
|    -> Select this repo                                      |
|    -> Main file: streamlit_app.py                         |
|    -> Click Deploy!                                       |
|                                                          |
|  Step 4: Get your online URL                              |
|    Format: https://xxxxx.streamlit.app                    |
+----------------------------------------------------------+
""")
    
    return github_url

def generate_report(gitee_url=None, github_url=None):
    report_path = os.path.join(BASE_DIR, "deployment_report.md")
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    report = f"""# Deployment Report
> Generated: {now}
> Project: {PROJECT_NAME} {PROJECT_VERSION}
> AI Stack: Qwen3.6-Plus + GLM-5 + MiniMax M2.7 (100% Domestic)

---

## Deployment Checklist

| # | Item | Status |
|---|------|--------|
| 1 | Git Environment | DONE |
| 2 | Python {sys.version.split()[0]} | DONE |
| 3 | streamlit_app.py entry point | READY |
| 4 | requirements.txt dependencies | READY |
| 5 | .streamlit/config.toml config | READY |
| 6 | .gitignore rules | READY |
| 7 | src/ source modules | PACKED |

## Deployment Package Structure

```
deploy_streamlit_cloud/
+-- streamlit_app.py          # Streamlit entry point
+-- requirements.txt          # Python dependencies
+-- deploy_to_gitee.py         # This deployment script
+-- .streamlit/
|   +-- config.toml           # Streamlit configuration
+-- src/                      # Source code modules
|   +-- app.py                # Main application
|   +-- cloud_config.py       # Cloud configuration
|   +-- data/                 # Data loading & processing
|   +-- models/               # Core algorithms (Agri-PC/ACML/CCP)
|   +-- utils/                # Utility functions
|   +-- visualization/        # Plotting & charts
|   +-- tabs/                 # UI page tabs
+-- .gitignore                # Git exclusion rules
+-- README.md                 # Project description
```

## Repository URLs

"""
    
    if gitee_url:
        report += f"""### Gitee Private Repository
- **URL**: {gitee_url}
- **Visibility**: LOCK (Private)
- **Status**: Ready to push

**Push command**:
```bash
cd deploy_streamlit_cloud
git push -u origin master
```

"""
    
    if github_url:
        report += f"""### GitHub Public Repository
- **URL**: {github_url}
- **Visibility**: Public
- **Status**: Ready to push

**Streamlit Cloud Deployment**:
1. Visit https://share.streamlit.io
2. Click "Deploy an app"
3. Select "From GitHub"
4. Choose this repository
5. Main file path: `streamlit_app.py`
6. Click "Deploy!"

"""
    
    report += """---

*Auto-generated by deploy_to_gitee.py*
*100% Domestic AI Tools: Qwen3.6-Plus / GLM-5 / MiniMax M2.7*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Report generated: {report_path}")
    return report_path

def main():
    print("""
+==============================================================+
|  [Agri] Agricultural Futures Pricing System - Deploy Tool      |
|  AI Stack: Qwen3.6-Plus + GLM-5 + MiniMax M2.5 (Domestic)      |
|  Target: Gitee(Private) / Streamlit Cloud(Public)              |
+==============================================================+
""")
    
    mode = "private"
    for arg in sys.argv[1:]:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1].strip()
    
    log(f"Deployment mode: {'Private+VPS' if mode=='private' else 'Public+StreamlitCloud'}")
    log(f"Working dir: {DEPLOY_DIR}")
    
    if not check_prerequisites():
        sys.exit(1)
    if not init_git_repo():
        sys.exit(1)
    if not add_and_commit():
        sys.exit(1)
    
    gitee_url = None
    github_url = None
    
    if mode == "private":
        gitee_url = show_private_guide()
    elif mode == "public":
        github_url = show_public_guide()
    else:
        log(f"Unknown mode: {mode}, defaulting to private", "WARN")
        gitee_url = show_private_guide()
    
    generate_report(gitee_url, github_url)
    
    print("""
+==============================================================+
|                    DEPLOYMENT PREP COMPLETE!                   |
+================================================--------------+
|  Next steps depending on your choice:                        |
|                                                              |
|  [PRIVATE MODE]:                                             |
|    1. Create private repo on Gitee                           |
|    2. Run: git push -u origin master                         |
|    3. Buy VPS (Alibaba/Tencent Cloud ~50RMB/mo)              |
|    4. Clone, install deps, run streamlit                     |
|                                                              |
|  [PUBLIC MODE]:                                              |
|    1. Create public repo on GitHub                           |
|    2. Run: git push -u github main                           |
|    3. Connect on share.streamlit.io -> Deploy                |
|                                                              |
|  See details: deployment_report.md                           |
+==============================================================+
""")

if __name__ == "__main__":
    main()
