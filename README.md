# Quantitative Finance Lectures

A free, open, citation-backed curriculum in quantitative finance — mathematics, statistics, stochastic calculus, derivatives pricing, portfolio theory, and machine learning applied to markets. Built as a static site with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages.

Live site (after deployment): `https://jsbldquant04.github.io/qf-lectures/`

## What's here

- `mkdocs.yml` — site configuration, theme, navigation, and Markdown extensions (MathJax, syntax highlighting, admonitions, etc.)
- `docs/` — all lecture content, organized by module (`prerequisites/`, `financial-foundations/`, `statistics/`, `portfolio-risk/`, `derivatives/`, `stochastic-finance/`, `trading/`, `advanced/`, `machine-learning/`, `research-lab/`, `references/`)
- `docs/_template/lecture-template.md` — the reusable template every new lecture should be copied from
- `.github/workflows/deploy.yml` — GitHub Actions workflow that builds and deploys the site to GitHub Pages on every push to `main`
- `requirements.txt` — Python dependencies for building the site locally or in CI

## Running locally in VS Code

### 1. Prerequisites

- Python 3.9+ installed
- [VS Code](https://code.visualstudio.com/) with the Python extension (optional but recommended)
- Git

### 2. Clone and set up a virtual environment

Open a terminal in VS Code (`` Ctrl+` `` / `` Cmd+` ``) and run:

```bash
git clone https://github.com/jsbldquant04/qf-lectures.git
cd qf-lectures

python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Serve the site locally with live reload

```bash
mkdocs serve
```

Then open **http://127.0.0.1:8000** in your browser. `mkdocs serve` watches `docs/` and `mkdocs.yml` and hot-reloads on save — this is the normal way to write and preview lectures while editing in VS Code.

### 4. Build a static version (optional, for a local sanity check)

```bash
mkdocs build --strict
```

This writes the built site to `site/` and fails loudly (`--strict`) on broken internal links or malformed nav entries — the same check the GitHub Actions workflow runs before deploying, so it's worth running before every push.

## Deploying to GitHub Pages

### One-time setup

1. Create a new **public** GitHub repository named `qf-lectures` (or update `site_url`/`repo_url`/`repo_name` in `mkdocs.yml` and the badge links in this README if you use a different name).
2. Push this project to it:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Quantitative Finance Lectures"
   git branch -M main
   git remote add origin https://github.com/jsbldquant04/qf-lectures.git
   git push -u origin main
   ```

3. In the repository on GitHub, go to **Settings → Pages**, and under **Build and deployment → Source**, select **Deploy from a branch**, then choose the **`gh-pages`** branch and **`/ (root)`** folder. (The first successful run of the Actions workflow below will create the `gh-pages` branch automatically — you can set this after the first push and workflow run.)
4. In **Settings → Actions → General → Workflow permissions**, ensure **"Read and write permissions"** is selected, so the deployment workflow (which uses `mkdocs gh-deploy`) is allowed to push to `gh-pages`.

### Every subsequent update

Just push to `main`:

```bash
git add .
git commit -m "Add lecture: <title>"
git push
```

The `.github/workflows/deploy.yml` workflow will automatically:

1. Check out the repository
2. Install Python and the dependencies in `requirements.txt`
3. Run `mkdocs build --strict` to catch broken links/config errors
4. Run `mkdocs gh-deploy --force --clean` to build and publish to the `gh-pages` branch

Within a minute or two of the workflow finishing, the update is live at:

```
https://jsbldquant04.github.io/qf-lectures/
```

You can also trigger the workflow manually from the **Actions** tab (it's configured with `workflow_dispatch`).

## Adding a new lecture

1. Copy `docs/_template/lecture-template.md` into the appropriate module folder and rename it (e.g. `docs/derivatives/black-scholes.md`).
2. Fill in all twelve sections (learning objectives, intuition, ... references).
3. Add the new file to the `nav:` section of `mkdocs.yml` under the correct module.
4. Update the corresponding row in `docs/curriculum.md` from "Planned" to "Published," linking to the new page.
5. Add any new citations to `docs/references/index.md`.
6. Run `mkdocs serve` locally to proof the page (math rendering, code highlighting, links) before pushing.

## License / usage note

This is an educational project. Code examples are pedagogical demonstrations only and are not investment advice. See `docs/references/index.md` for the citation policy.
