# Paris Ensemble

Plan a trip to Paris that a whole group can agree on. Advanced Machine Learning, Project 13: Planning Group Trips with LLM Agents.

Each traveler describes their wishes in their own words, in any language. One LLM agent per traveler reads those wishes and scores places. An optimizer builds the shared plan. The agents then object, a mediator decides which requests to grant, and the optimizer builds the plan again.

## How a group uses it

1. The organizer enters the password, chooses the dates, and sends the group one invite link, by copying it, by email or on WhatsApp. Email opens the sender's own mail app with the message written, so the app needs no mail server.
2. Each friend opens the link on their own phone and fills in only their own part. They need no password, because the link opens this one trip and nothing else. Saving gives them a personal link to come back with.
3. Nobody sees anyone else's answers. The group only sees who has answered.
4. When everyone has answered, planning starts by itself. If someone is late, the organizer can plan without them.
5. Everyone sees two plans, how well each fits them, and only anonymous numbers for the rest of the group. Each person rates and votes on their own, without knowing how each plan was made.
6. When the last vote is in, or the organizer closes the vote, the app picks the best plan: most votes, then the better rating, then the plan fairest to the group. Then it shows how each plan was made.

**The LLM never writes the plan in our method.** It returns profiles, scores, ids and weights. OR-Tools CP-SAT and a scheduler build the plan on real opening hours, and one validator checks every plan from every method.

## The five methods

| | Method | How it decides |
| --- | --- | --- |
| M1 | Average | Maximizes the sum of everyone's predicted utility |
| M2 | Least misery | Maximizes the least satisfied member first, then the sum |
| M3 | Fairness rule | Maximizes the sum while every member stays above a threshold (0.6, lowered by 0.05 until it can be met) |
| M4 | Agent negotiation | Starts from M1. Agents review the plan and object, a mediator picks requests and sets weights, and the optimizer solves again, for up to 4 reviews (3 in the app). A request is kept only if the group's predicted mean and lowest satisfaction stay within 0.02 of the start. The final plan is always one the agents reviewed. Falls back to M3 if it ends with someone below the threshold |
| M5 | Single LLM planner | One LLM call writes the whole plan. It is checked but not repaired. This is the baseline |

**Satisfaction** is the utility of a plan for a member divided by the best they could get alone (S\*), capped at 1. Stops a group could not really do earn nothing: closed at that time, too late in the day, past the pace they asked for, or a repeat.

**The learned part** is one small model per traveler. A straight line from profile score to agent score is fitted to that traveler's agent scores on about 60 places, and the agent's own score also counts for 25% on the places it scored. Add and remove actions train a ridge correction on PCA of the place embeddings, moving a target at most 1 point at a time.

A richer model that learned from the agent's scores through embeddings did worse on real LLM scores, so it was dropped. Most of the gain from feedback also comes from removed places staying out and added places staying in, not from updating the model.

**Settings were chosen on 12 development groups, never on the 40 test groups:** `compare_real_models.py` for the preference model, `compare_feedback_real.py` for how far feedback spreads, and `compare_negotiation.py` for what a request may cost.

**Simulated travelers** have hidden theme weights, price sensitivity, needs and one specific passion, such as Impressionist painting, which no theme can express. An LLM writes their wishes from this, and the system only ever sees those wishes and the form fields. Their true taste scores every plan.

## Run it

Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env               # then fill in OPENAI_API_KEY, APP_PASSCODE and ADMIN_PASSCODE
python scripts/smoke_llm.py        # the first real API test
streamlit run app.py
```

To try the app with no key and no cost, set `LLM_MOCK=1`. A rule based stand-in answers instead of the LLM, and the app shows a warning banner. Results made that way mean nothing.

`/?admin=1` opens the page for downloading the study answers as Excel. [DEPLOY.md](DEPLOY.md) explains how to put the app online.

To try the group flow on your own network, start with `streamlit run app.py --server.address 0.0.0.0` and open the app at your computer's network address, so the invite link works on phones on the same network. Alone, "Try it alone with an example group" fills in made up travelers and lets you rate for them.

## Commands

| Task | Command | Uses the API |
| --- | --- | --- |
| Tests and lint | `pytest` and `ruff check .` | No, they cannot spend money |
| Test the key, the models and the prompts | `python scripts/smoke_llm.py` | Yes |
| One real trip through all five methods | `python scripts/smoke_llm.py --trip` | Yes |
| Check the place data | `python scripts/check_places.py` | No |
| Check name matching | `python scripts/check_places.py --names` | Embeddings only |
| Make the sheet for checking places by hand | `python scripts/check_places.py --sample 30` | No |
| Stress test plans on real data | `LLM_MOCK=1 python scripts/check_plans.py --trips 40` | No |
| A group of three in separate browsers, checking privacy at every step (needs `pip install playwright`, and the app running on port 8599 with password `test`) | `python scripts/e2e_group.py` | Only if the app runs with the real key |
| Download places again | `python -m src.data.fetch_osm`, then `python scripts/build_places.py --stage base` | No |
| Score places with the LLM | `python scripts/build_places.py --stage enrich` | Yes |
| Write the simulated groups | `python scripts/run_experiments.py groups` | Yes |
| Pilot, then the full main experiment | `python scripts/run_experiments.py E1 --limit 4 --out results/pilot`, then `python scripts/run_experiments.py E1` | Yes |
| Feedback experiment | `python scripts/run_experiments.py E3` | Yes |
| Figures, tables and statistics | `python scripts/make_figures.py` | No |

Experiments continue where they stopped, and LLM answers are cached, so a rerun only pays for what is missing. Each run saves its settings, prompt versions and data version in `run_info.json`.

## Where the data comes from

| What | Source | How far to trust it |
| --- | --- | --- |
| Places, coordinates, cuisines, stars, addresses | OpenStreetMap through Overpass, November 2026 | Real, community maintained |
| The 154 attractions | Reviewed by hand against the Wikidata popularity ranking, `data/curated_attractions.yaml`. Leaves out closed places (Centre Pompidou, Tour Montparnasse deck, Palais de la Découverte), places with no verifiable hours, and duplicates | Reviewed by hand |
| Opening hours | OpenStreetMap, or checked on the web and written in the curated file. Resolved for the real date, so seasons and closed days count. None are guessed | Good, but check before you go |
| Vegetarian options | Only what OpenStreetMap states. A group with a vegetarian eats only at those places. We tried asking the LLM and it was unreliable | Good, but menus change |
| Ticket prices | 15 main sites checked on the web in November 2026; the others are marked as estimates | Estimates, the higher non-EU rate where there are two |
| Hotel prices | A typical price by star rating | Estimates |
| Meal prices | 18, 35 or 80 EUR by a price level the LLM picks, and it mostly says mid range | Rough estimates |
| Themes of each place | LLM, from the name, the description and the tags | Checked on clear cases by `check_places.py` |
| Photos | Wikimedia Commons, credited in the app | Real |

Fonts: Satoshi is loaded from [Fontshare](https://www.fontshare.com/fonts/satoshi), and Cinzel is included under the SIL Open Font License.

Prices are planning estimates, not quotes. Evenings after dinner are left unplanned, and the app says so.

## Layout

```
app.py                 entry point; copies Streamlit secrets into the environment
config.yaml            every constant in one place
prompts/               the 8 LLM prompts, each with a version
data/                  the place file, embeddings, zones, and the curated list
src/llm.py             every OpenAI call: structured output, cache, cost log
src/dev_mock.py        the offline stand-in, active only with LLM_MOCK=1
src/data/              downloads, geography, opening hours, place loading, name matching
src/agents/            traveler agent and mediator
src/consensus/         negotiation (M4) and the single LLM planner (M5)
src/planner/           optimizer, scheduler, validator, satisfaction
src/prefs/             the per traveler preference model
src/pipeline.py        the one path from wishes to a checked plan, used by the app and the experiments
src/jobs.py            planning in background threads, so a locked phone loses nothing
src/store.py           SQLite, Excel export, Hugging Face backup
src/eval/              simulated travelers, metrics, experiment helpers
scripts/               building data, checks, experiments, figures
ui/                    pages, views and theme of the app
tests/                 run with pytest; a dummy key is forced, so they cannot spend money
```

## What the validator checks

| Code | Problem |
| --- | --- |
| V1 | A place that does not exist |
| V2 | A place used twice |
| V3 | Closed at that time |
| V4 | Not enough time to travel to the next stop |
| V5 | The day runs too late |
| V6 | A meal is missing or badly timed |
| V7 | Over the lowest budget in the group |
| V8 | Ignores an avoid, a vegetarian need or a mobility need |
| V9 | Wrong number of days |
| V10 | More visits than the group's pace allows |

## Sources for facts in the curated list (checked November 2026)

Prices: [Louvre](https://abc13.com/post/louvre-museum-paris-hikes-ticket-prices-most-european-tourists-what-know/18401055), [Eiffel Tower](https://www.toureiffel.paris/en/preparing-your-visit/rates-and-visiting-conditions), [Musée d'Orsay](https://www.musee-orsay.fr/en/visit/rates), [Musée de l'Orangerie](https://www.musee-orangerie.fr/en/articles/rates-257), [Centre des monuments nationaux](https://www.monuments-nationaux.fr/en/adult-groups-and-tourism-professionals/what-is-advance-ticketing/rates-2026), [Catacombs](https://www.artvisitguide.com/paris-catacombs-tickets-guide), [Palais Garnier](https://www.operadeparis.fr/visites/palais-garnier), [Rodin Museum](https://thebettervacation.com/fr/Paris/mus%C3%A9e-rodin-paris), [Picasso Museum](https://thebettervacation.com/fr/Paris/billets-mus%C3%A9e-picasso-paris), [Fondation Louis Vuitton](https://lesfrenchiestravel.com/things-to-do-in-paris-2026/things-to-do-in-paris-fondation-louis-vuitton), [Musée Grévin](https://tarifs.org/musee-grevin-tarif), [Bateaux-Mouches](https://www.cometoparis.com/catalog/cruise-tour-bateaux-mouches-9000529), [metro fare](https://www.sortiraparis.com/en/news/in-paris/articles/337462-navigo-pass-and-metro-train-tickets-new-fares-expected-in-january-2026) and [metro guide](https://parisbytrain.com/paris-metro), [hotels](https://lavie.maison/blog/hotel-paris-prices), [meals](https://france-hotel-guide.com/en/blog/daily-cost-food-paris) and [more on meals](https://www.eatingeurope.com/blog/cost-of-food-in-paris-per-day).

Closures: [Centre Pompidou](https://www.sortiraparis.com/en/what-to-visit-in-paris/exhibit-museum/articles/241921-closing-of-the-centre-pompidou-the-museum-will-not-reopen-until-2030), [Palais de la Découverte](https://www.palais-decouverte.fr/sites/default/files/documents/2026-06/SaisonCulturelle%2026-27.pdf), [Montparnasse Tower](https://www.thebettervacation.com/paris/montparnasse-tower-tickets).

Hours: [Gustave Moreau](https://parisjetaime.com/musee-monument-paris/71431/Musee-national-Gustave-Moreau), [Coulée Verte](https://parisjetaime.com/musee-monument-paris/71237/Coulee-verte-Rene-Dumont), [Chapel of the Miraculous Medal](https://www.sortiraparis.com/arts-culture/histoire-patrimoine/articles/349509-chapelle-de-la-medaille-miraculeuse-une-chapelle-discrete-qui-cache-un-lieu-de-pelerinage-mondial), [Alexander Nevsky](https://parisjetaime.com/culture/cathedrale-saint-alexandre-nevsky-p3511), [Saint-Germain l'Auxerrois](https://parisjetaime.com/culture/eglise-saint-germain-l-auxerrois-p1229), [Saint-Roch](https://parisjetaime.com/culture/paroisse-saint-roch-p1130), [Notre-Dame-des-Victoires](https://dioceseparis.fr/-notre-dame-des-victoires-.html), [Saint-Paul-Saint-Louis](https://parisjetaime.com/culture/eglise-saint-paul-saint-louis-du-marais-p1247).
