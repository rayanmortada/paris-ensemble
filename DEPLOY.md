# Deploying Paris Ensemble

The app runs on Streamlit Community Cloud at a public link. Starting a trip needs the password. A friend needs only the invite link, which opens that one trip and nothing else. It takes about 20 minutes.

## 1. Before you start

You need:

- a GitHub account
- a working OpenAI key, with a monthly spending limit set in the OpenAI dashboard (Settings, Limits). This is your real protection if the password leaks
- optionally a free Hugging Face account, to keep the study answers when the app restarts (step 5)

Make sure the place data was built with the real API. `data/build_info.json` must say `"enrichment": "openai"`, and this must pass:

```bash
python scripts/check_places.py
```

## 2. Put the project in its own private GitHub repository

The project needs its own repository, and your home folder is already a git repository. So run `git init` inside the project folder, or `git add` would add these files to the one in your home folder.

```bash
cd "group-trip-planner"
git init -b main
git config user.name                # check that this is the name you want on the commits
git add .
git status                          # .env and data/raw must NOT be listed
git commit -m "Paris Ensemble"
gh repo create paris-ensemble --private --source=. --push
```

These files must be in the commit, because the app reads them at start: `data/paris_places.json`, `data/embeddings.npy`, `data/zones.json`, `data/build_info.json`.

The repository can be public. Its only font file is Cinzel, under the SIL Open Font License (`static/fonts/Cinzel-OFL.txt`). Satoshi is loaded from Fontshare, its maker's own servers, as its free licence asks, so its files are not in the repository.

## 3. Create the app

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub. If the repository is private, allow access to private repositories when asked.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Choose the repository, branch `main`, main file `app.py`.
4. Open **Advanced settings**. Set Python to **3.12**, which is the version the tests ran on with these exact package versions.
5. Paste this into **Secrets**, with your own values:

```toml
OPENAI_API_KEY = "sk-..."
APP_PASSCODE = "the password for whoever starts a trip"
ADMIN_PASSCODE = "a different one, only for you"
MODEL_FAST = "gpt-5.4-mini"
MODEL_STRONG = "gpt-5.5"
EMBED_MODEL = "text-embedding-3-small"
# PUBLIC_URL = "https://your-app.streamlit.app"   # only if invite links come out with the wrong address
# Only with step 5:
# HF_TOKEN = "hf_..."
# FEEDBACK_DATASET_REPO = "your-name/paris-ensemble-study"
```

Without `APP_PASSCODE` the app stays locked and says so. It never opens without a password.

6. Click **Deploy**. The first build takes a few minutes.

## 4. Make the link public

An app from a private repository may start as private, meaning only invited accounts can open it. If testers are asked to sign in to Streamlit, open the app's **Settings**, then **Sharing**, and choose that everyone with the link can view it. Starting a trip still needs your password.

## 5. Keep the answers (recommended before the user study)

The free hosting wipes its disk whenever the app restarts, which would lose the ratings. To keep them:

1. On Hugging Face, create a **private** dataset, for example `your-name/paris-ensemble-study`.
2. Create an access token with write permission.
3. Add `HF_TOKEN` and `FEEDBACK_DATASET_REPO` to the app's Secrets.

The database is then copied to that dataset at most every 2 minutes and after every rating, and restored at start.

## 6. Check it

On a laptop and on a phone:

1. Open the link. A wrong password must be refused.
2. Enter the password and start a trip. The invite link must show your app's public address. If it does not, add `PUBLIC_URL` to the Secrets.
3. Send yourself the invite and open it on your phone with no password. Fill in your part there and your own part on the laptop. Neither page may show the other's answers.
4. When everyone has answered, planning starts by itself. It should take about one to two minutes.
5. Vote on both devices. When the last vote is in, both pages must move on to the winning plan.
6. Open `<your link>/?admin=1`, enter `ADMIN_PASSCODE`, and download the Excel file. The votes must be in it.
7. No yellow warning banner should appear. A banner means the app is using the offline stand-in (`LLM_MOCK`) or data built by it.

## What it costs

Measured from the token counts of real runs, at the prices listed for these models in November 2026 (gpt-5.5 $5 and $30, gpt-5.4-mini $0.75 and $4.50, per million input and output tokens; from third party price lists, so check OpenAI's page):

| Action | Cost |
| --- | --- |
| One group's two plan comparison | about $0.12 for 3 people, $0.20 for 5 |
| Comparing all five methods as well | about $0.30 more |
| One round of changing a plan | about $0.10 |

The app allows 3 trips per organizer visit and 40 per day (`app` in `config.yaml`), so even a very busy day stays around $30. Invite links cannot start the paid comparison of all five methods: only the organizer can. Someone who gets hold of an invite could still answer in a free seat, and the last answer starts one plan for that trip, so the most a leaked invite can cost is about $0.20.

## Updating

Every `git push` to `main` redeploys the app. Changing the Python version means deleting and recreating the app, and pasting the Secrets again.

The interface styling depends on Streamlit's page structure, which is why `requirements.txt` pins exact versions. Look at the app on a phone after any upgrade.

## Limits of the free hosting

Streamlit's documentation gives between 690 MB and 2.7 GB of memory and up to 2 CPU cores, and these can change without notice. Apps go to sleep when unused, and the next visitor wakes them with a button.

If it becomes too slow, Render is the fallback: create a Web Service from the repository with build command `pip install -r requirements.txt` and start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, add the same settings as environment variables, and set `PYTHON_VERSION` to `3.12.13`. Add a disk and point `DATA_DIR` at it if you want the database to survive restarts without Hugging Face.

## If something goes wrong

| What you see | Cause |
| --- | --- |
| "This app has no password set" | `APP_PASSCODE` is missing from the Secrets |
| Invite links show the wrong address | Add `PUBLIC_URL` to the Secrets |
| A friend lost their personal link | The organizer removes them under "Change the group", and they join again with the invite, before planning |
| "The OpenAI key is missing or wrong" | Check `OPENAI_API_KEY`. Keys pasted into a chat or a public place can get disabled by OpenAI |
| "OpenAI refused the request because of a spending or rate limit" | The account ran out of credit or reached its limit |
| "The OpenAI model name in the settings is not available" | Run `python scripts/smoke_llm.py` locally. It lists the models your key can use |
| The build fails while installing packages | Check that Python is set to 3.12 |
