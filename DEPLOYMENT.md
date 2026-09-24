# Deployment — Streamlit Community Cloud

GravityHunter can be deployed directly from GitHub on Streamlit Community Cloud. The bundled HDF5 data are small enough to live in the repository, so the deployed app does not need a runtime downloader or credentials.

## 1. Create a safe UI branch

From the repository root:

```bash
git checkout main
git pull
git checkout -b ui-polish
```

Copy the UI-polish patch over the repository root, then inspect the changes:

```bash
git status
git diff
```

## 2. Verify the project before committing

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/validate_bundled_data.py
python -m streamlit run streamlit_app.py
```

The scientific backend should remain unchanged by the UI branch. The current verified build has 19 automated tests after the inspiral-ridge regression was added.

## 3. Commit the bundled HDF5 files

The deployment machine receives files from the GitHub repository. Therefore the real H1/L1 and template HDF5 files must be tracked by Git.

This UI/deployment patch removes the old `data/**/*.hdf5` ignore rule. The complete bundled data directory is only about 9 MB, with individual files around 1 MB, so Git LFS is not required for this project snapshot.

Check that the files exist:

```text
data/GW150914/H1.hdf5
data/GW150914/L1.hdf5
data/GW151226/H1.hdf5
data/GW151226/L1.hdf5
data/GW170104/H1.hdf5
data/GW170104/L1.hdf5
data/templates/GW150914_4_template.hdf5
data/templates/GW151226_4_template.hdf5
data/templates/GW170104_4_template.hdf5
```

Then stage them with the application changes:

```bash
git add .streamlit/config.toml .gitignore README.md DEPLOYMENT.md requirements.txt
git add streamlit_app.py gravityhunter/ui/merger_animation.py
git add data/GW150914 data/GW151226 data/GW170104 data/templates data/SHA256SUMS.txt
git status
```

Do not commit `.venv`, `__pycache__`, `.pytest_cache`, diagnostics, or partial `.hdf5.part` files.

## 4. Commit and push the branch

```bash
git commit -m "Polish analysis UI and prepare Streamlit deployment"
git push -u origin ui-polish
```

Review the branch on GitHub. If everything looks correct, merge it into `main` through a pull request or locally after final testing.

## 5. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/.
2. Sign in with GitHub and authorize access to the GravityHunter repository.
3. Click **Create app**.
4. Choose the existing GitHub repository.
5. Set:
   - **Repository:** `SaklainTheSpellbinder/GravityHunter`
   - **Branch:** `main` after the UI branch is merged
   - **Main file path:** `streamlit_app.py`
6. Open **Advanced settings**.
7. Select **Python 3.13** to match the locally validated environment.
8. No secrets are required for the bundled-data build.
9. Optionally choose a custom `*.streamlit.app` subdomain.
10. Click **Deploy**.

Community Cloud installs `requirements.txt`, copies the repository, and runs the Streamlit entrypoint from the repository root.

## 6. Verify the deployed application

After deployment, check at least:

- GW150914, GW151226, and GW170104 load without missing-file errors.
- Event Detection recovers the expected candidates.
- Detector Coincidence reports the expected H1/L1 results.
- Inspiral Diagnostics produces the bundled ridge results.
- Event Timeline renders and the GPS clock advances correctly.
- **Auto focus** and **Full record** change only visualization framing.

If the app says the HDF5 files are missing, confirm that Git actually tracks them:

```bash
git ls-files data
```

The output must include the `.hdf5` files, not just `data/README.md`.

## 7. Updating the deployed app

Streamlit Community Cloud watches the configured GitHub branch. Push a new commit to that branch and the deployed app is updated automatically in most cases.

For backend changes, run the full test suite locally before pushing.

## Cost

Streamlit Community Cloud provides a free deployment tier for Streamlit applications. GravityHunter does not require paid external APIs, a database, or secrets for the bundled-data configuration.

Official deployment documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud
