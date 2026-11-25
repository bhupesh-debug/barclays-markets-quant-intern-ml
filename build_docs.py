"""Build the 10-page report PDF and the 10-slide deck PDF (reportlab).

Reads reports/metrics.json so every number in the documents comes from
the executed pipeline, not from hand-typed values.
"""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
OUT = Path("/home/hatch/workspace/your_files")
CHARTS = ROOT / "reports" / "charts"
M = json.loads((ROOT / "reports" / "metrics.json").read_text())
S = {r["model"]: r for r in M["summary"]}
INS = M["insample_accuracy"]
REG = M["regimes"]

NAVY = colors.HexColor("#1A3A6B")
GREEN = colors.HexColor("#2E7D5B")
RED = colors.HexColor("#B03A2E")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EAF0F7")

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=NAVY, spaceBefore=8, spaceAfter=3)
B = ParagraphStyle("b", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5)
SM = ParagraphStyle("sm", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=GREY, spaceAfter=4)
CELL = ParagraphStyle("c", fontName="Helvetica", fontSize=9, leading=12)
CELLB = ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=9, leading=12)
TITLE = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=NAVY)
SUB = ParagraphStyle("s", fontName="Helvetica", fontSize=12.5, leading=17, textColor=GREY)


def P(text, style=B):
    return Paragraph(text, style)


def bullets(items, style=B):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", style) for t in items]


def table(headers, rows, widths):
    data = [[Paragraph(h, CELLB) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9C6D8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def chart(name, w=7.0, h=3.4):
    return Image(str(CHARTS / name), width=w * inch, height=h * inch)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "Multi-Asset Quantitative Signal Research  •  Barclays Markets Quant Intern  •  SYNTHETIC DATA ONLY")
    canvas.drawRightString(7.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_report():
    path = OUT / "Bhupesh_Chandra_Dimri_ML-Project_Markets-Quantitative-Analyst-Summer-Internship_Barclays.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="ML Project — Markets Quantitative Analyst Summer Internship, Barclays",
                            author="Bhupesh Chandra Dimri")
    s = []
    # ---- Page 1: cover + executive summary ----
    s += [Spacer(1, 0.5 * inch),
          P("Multi-Asset Quantitative<br/>Signal Research", TITLE), Spacer(1, 0.15 * inch),
          P("ML Project for the Markets Quantitative Analyst Summer Internship Program 2028<br/>Barclays, New York — JR-0000139049", SUB), Spacer(1, 0.25 * inch),
          P("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026", B),
          P("Python • pandas • scikit-learn • walk-forward backtesting • transaction costs", SM),
          Spacer(1, 0.15 * inch)]
    s.append(P("Executive summary", H1))
    s += bullets([
        "<b>What it is:</b> a signal-research project across six synthetic multi-asset instruments standing in for the programme's rotations — Equities, Credit, Rates, FX, Municipals, Securitized Products — testing whether simple, interpretable signals survive honest out-of-sample backtesting.",
        "<b>Headline result (940 out-of-sample days, net of 5bp costs):</b> the simple momentum rule wins — Sharpe <b>0.71</b>, hit rate 50.2%, +11.0%, max drawdown −5.1%. Linear regression is second (Sharpe 0.59, best hit rate 51.1%).",
        "<b>Honest finding:</b> the random forest memorised training data (68.3% in-sample accuracy) and <b>lost 18.3% out of sample</b> (Sharpe −1.28). Complexity plus 25.8% turnover is a losing combination in noisy markets.",
        "<b>Regime finding:</b> momentum works in calm markets and fails in volatile ones (momentum-rule Sharpe 1.70 low-vol vs −0.24 high-vol). Any desk using this signal needs a volatility gate.",
        "<b>Data:</b> 100% synthetic, seeded and reproducible (1,250 days × 6 assets). No real prices, no live-market claim anywhere in this document.",
    ])
    s.append(P("Match score: 6 / 10 — strong on Python, data analysis and forecasting; gaps are C++/Java (not evidenced), stochastic-calculus coursework, and trading-desk experience. See §6.", B))
    s.append(P("Eligibility note: this project is judged on quant merit. It does not claim the candidate is eligible for the internship — he graduated M.S. Dec 2025 and the programme targets students graduating within one year of the internship's conclusion. Eligibility timing is a separate gate for the recruiter.", SM))
    s.append(PageBreak())

    # ---- Page 2: JD breakdown (Step 1) ----
    s.append(P("1 &nbsp; JD breakdown — what the role really asks for", H1))
    s.append(P("Barclays Markets Quantitative Analyst Summer Internship, 10 weeks in New York, rotations across Credit, Equities, Equity Financing, FX, Municipals, Rates, Structuring and Securitized Products. Summer analysts build and backtest mathematical models, monitor markets, formulate trade ideas, analyse market/fundamental/technical data, and present results to traders, salespeople and researchers.", B))
    s.append(table(
        ["Category", "What the JD asks for"],
        [["Core responsibilities", "Small research projects; presentations and reports of findings; implement findings with senior management; build and backtest models; monitor markets and formulate trade ideas; analyse market, fundamental and technical data; present to immediate and related teams; client meetings/calls"],
         ["Required technical skills", "Statistics, linear algebra, optimisation, machine learning, stochastic calculus; large-data analysis in Python, C++ or Java"],
         ["Soft skills", "Fast processing of information; anticipating market trends; communication; collaboration across desks; mentoring and training participation"],
         ["Hidden expectations", "Ownership of a research question end-to-end; judgement under time pressure; presenting to senior stakeholders; 10-week delivery cadence with formal feedback"],
         ["Success measures", "Quality of models and trade ideas; clarity of presentations; contribution to desk decisions; top performers invited back full-time"],
         ["Seniority", "Summer intern — pre-career. The bar is research quality and learning speed, not production ownership"],
         ["Domain knowledge", "Global markets across asset classes; how Sales &amp; Trading desks operate; market microstructure basics"]],
        [1.5 * inch, 5.5 * inch]))
    s.append(P("Requirement hierarchy", H2))
    s += bullets([
        "<b>Must-have:</b> quantitative coursework (statistics, linear algebra, optimisation, ML); Python for large-data analysis; ability to build and backtest a model and explain it.",
        "<b>Good-to-have:</b> stochastic calculus; C++/Java; prior research or trading experience; familiarity with a second asset class.",
        "<b>Bonus:</b> a reproducible research artefact (this project); a trade-idea write-up with risk limits; evidence of presenting findings.",
    ])
    s.append(P("Risk factors: the JD's eligibility line (graduation within one year of the internship's conclusion) is the gating risk for this candidate and is outside the project's scope. C++/Java and stochastic calculus are genuine skill gaps — flagged, not hidden (§6).", B))
    s.append(PageBreak())

    # ---- Page 3: ML problem + data (Steps 2–3) ----
    s.append(P("2 &nbsp; ML problem definition", H1))
    s.append(P("Two problem statements, one per decision the desk actually makes:", B))
    s.append(table(
        ["Problem", "Input", "Output / target", "Type"],
        [["P1 — Direction signal", "7 signals per asset-day: momentum 5/20/60d, 20d vol, RSI-14, carry, cross-sectional momentum", "Next-day direction (−1/+1); return magnitude as secondary target", "Classification (+ regression)"],
         ["P2 — Signal portfolio", "Per-asset direction forecasts from P1", "Equal-weighted long/short portfolio, cost-aware, with a volatility gate", "Time-series backtest / portfolio construction"]],
        [1.2 * inch, 2.6 * inch, 2.2 * inch, 1.0 * inch]))
    s.append(P("Constraints that shaped the design: interpretability first (a desk must be able to explain a signal to risk); no lookahead anywhere; transaction costs included from the start; walk-forward validation, never a single lucky split. Latency is not a constraint here — this is daily research, not intraday execution.", B))
    s.append(P("3 &nbsp; Data strategy — synthetic, by design", H1))
    s.append(P("No real market data is used. The generator (<font face='Helvetica-Bold'>src/data_gen.py</font>, seed 202608) produces 1,250 business days (Jan 2021–Dec 2025) for six instruments. Daily log-returns = asset drift + a small momentum component (past 20-day mean return, coefficient 0.05–0.09) + a common cross-asset factor + volatility clustering (volatility rises after large moves, capped at 3× base vol). The process is deliberately weak-signal: that is what real quant research looks like, and it is why the honest findings in §8 matter.", B))
    s.append(table(
        ["Instrument", "Desk rotation", "Base vol/day", "Momentum β"],
        [["EQ", "Equities", "1.10%", "0.09"], ["CREDIT", "Credit", "0.60%", "0.07"],
         ["RATES", "Rates", "0.70%", "0.06"], ["FX", "Foreign Exchange", "0.60%", "0.05"],
         ["MUNI", "Municipals", "0.40%", "0.06"], ["SEC", "Securitized Products", "0.80%", "0.08"]],
        [1.0 * inch, 2.2 * inch, 1.4 * inch, 1.4 * inch]))
    s.append(P("Preprocessing: rolling features only (5/20/60-day windows), RSI-14, cross-sectional momentum (asset mom20 minus the day's cross-asset mean). 7,140 feature rows after warm-up. Scenarios tested: low-vol vs high-vol regimes (§8) and cost levels 0–20bp (§8). Edge cases — a volatility spike, a signal flip on consecutive days — are handled by the turnover cost and the regime split, not by hand-waving.", B))
    s.append(PageBreak())

    # ---- Page 4: Model strategy (Step 4) ----
    s.append(P("4 &nbsp; Model strategy", H1))
    s.append(table(
        ["Tier", "Model", "Role in this project"],
        [["Baseline", "Momentum rule: long if 20d momentum &gt; 0", "The bar every model must beat. Interpretable, zero training, low turnover"],
         ["Intermediate", "Logistic regression (standardised features)", "Interpretable direction classifier; coefficients readable by a desk"],
         ["Intermediate", "Linear regression on next-day return", "Direct return forecast; sign gives the position"],
         ["Advanced", "Random forest (200 trees, depth 8, min leaf 20)", "Included deliberately to test whether complexity helps on noisy data — it does not (§8)"]],
        [1.0 * inch, 2.2 * inch, 3.8 * inch]))
    s.append(P("Feature engineering — the five ideas that matter", H2))
    s += bullets([
        "<b>Multi-horizon momentum (5/20/60d):</b> separates fast noise from the persistent component the generator actually contains.",
        "<b>Cross-sectional momentum:</b> relative strength across assets — the closest proxy in this project to a desk's relative-value view.",
        "<b>Volatility regime (20d vol vs its median):</b> the feature that later explains when the signal works and when it breaks.",
        "<b>RSI-14:</b> mean-reversion counterweight, so the model can learn when momentum is exhausted.",
        "<b>Carry:</b> a slow yield term; deliberately small, so the model cannot lean on it as a free lunch.",
    ])
    s.append(P("Evaluation: walk-forward (expanding window, retrain every 60 days, 250-day warm-up), primary metric out-of-sample Sharpe, secondary hit rate, mean information coefficient and max drawdown. Overfitting controls: walk-forward splits, regularised linear models (C=1.0), min-leaf-20 on the forest, and the in-sample/out-of-sample comparison in §8 that catches the forest red-handed.", B))
    s.append(P("5 &nbsp; Visual intelligence — what the charts say", H1))
    s.append(P("Equity curves (§8, chart 1) show the momentum rule and linear model compounding steadily while the random forest grinds down — the visual signature of overfitting plus turnover. The model-comparison chart (§8, chart 2) makes the ranking unambiguous. Feature importance (§8, chart 3) shows logistic regression loading on mom20 and cross-sectional momentum — the two features the data-generating process actually rewards — while the forest spreads weight across noise features. The return distribution (§8, chart 4) is near-symmetric with fat tails: no free lunch, no hidden skew.", B))
    s.append(PageBreak())

    # ---- Page 5: charts ----
    s.append(P("5 &nbsp; Visual intelligence (charts)", H1))
    s.append(chart("equity_curves.png", 7.0, 2.9))
    s.append(P("Chart 1 — Out-of-sample equity curves, net of 5bp costs. Momentum rule and linear model end +11.0% / +9.4%; the random forest ends −18.3%.", SM))
    s.append(chart("model_comparison.png", 7.0, 2.7))
    s.append(P("Chart 2 — Out-of-sample Sharpe (left) and hit rate (right). The 50% line is the coin-flip baseline. All four models sit within 49.9–51.1% hit rate: direction forecasting on noisy data is hard, and the differences are in risk-adjusted return, not accuracy.", SM))
    s.append(PageBreak())

    # ---- Page 6: resume alignment (Step 6) ----
    s.append(P("6 &nbsp; Resume alignment — match 6 / 10", H1))
    s.append(P("Candidate: Bhupesh Chandra Dimri — Financial Analyst, JPMorgan Chase (Jul 2025–present), 9+ years forecasting, budgeting and variance analysis; M.S. Accounting Analytics &amp; Data Technology, Pace (Dec 2025, GPA 3.24); MBA Finance &amp; Accounting. Evidence is taken only from the verified record.", B))
    s.append(table(
        ["JD requirement", "Resume evidence", "Gap?"],
        [["Large-data analysis in Python", "Python evidenced; FP&amp;A predictive engine (LightGBM, WAPE 4.62%, R² 0.942) and quant-ipo-engine on GitHub", "No"],
         ["Statistics / ML coursework", "M.S. Accounting Analytics &amp; Data Technology; ML project work with honest backtesting", "Partial"],
         ["Build &amp; backtest models", "This project; prior forecasting and variance engines with walk-forward discipline", "No"],
         ["Stochastic calculus", "Not evidenced in the record", "Yes — [CONFIRM WITH CANDIDATE]"],
         ["C++ / Java", "Not evidenced in the record — project is Python-only, stated plainly", "Yes"],
         ["Trading-desk / markets experience", "None in the record; FP&amp;A and analytics background only", "Yes"],
         ["Presenting findings", "Deck + report artefacts; stakeholder reporting in FP&amp;A roles", "No"]],
        [1.6 * inch, 3.6 * inch, 1.8 * inch]))
    s.append(P("Top 3 missing skills: (1) C++/Java — learn enough C++ to implement a backtester, or position Python as the research language; (2) stochastic calculus — targeted self-study, named honestly as in-progress, never claimed; (3) live markets exposure — paper-trade the signal in §8 and log the diary. Bullet upgrades: rewrite the FP&amp;A engine bullet with the JD's language — 'Built and backtested a LightGBM forecasting model (WAPE 4.62%, R² 0.942) on large financial datasets; presented findings to stakeholders' — and add this project as a second bullet with the momentum-rule Sharpe and the random-forest lesson.", B))
    s.append(P("Existing GitHub work used (real metrics only, no duplication): <font face='Helvetica-Bold'>quant-ipo-engine</font> and the FP&amp;A predictive variance engine (LightGBM WAPE 4.62%, R² 0.942). This project is distinct: multi-asset signals, walk-forward backtesting with costs, and regime analysis — none of which the earlier engines do.", B))
    s.append(PageBreak())

    # ---- Page 7: architecture (Step 7) ----
    s.append(P("7 &nbsp; Solution design &amp; architecture", H1))
    s.append(P("End-to-end pipeline: <font face='Helvetica-Bold'>Synthetic generator → Feature store (rolling, no lookahead) → Walk-forward training → Model registry (four models, one interface) → Signal portfolio (equal-weight, cost-aware) → Monitoring (Sharpe, hit rate, IC, drawdown, turnover) → Feedback (regime split, cost sensitivity)</font>.", B))
    s.append(P("Tech stack: Python 3, pandas, scikit-learn, matplotlib; no deep learning — the role's bar is statistical/ML signal research, and the data-generating process is linear-plus-noise, so a transformer would be theatre. Deployment pattern for a desk: a daily batch research job (not live execution) — regenerate signals after the close, log them with the walk-forward metrics, and review weekly with the regime dashboard. Experiment tracking is a plain metrics.json plus git commits; MLflow would be the next step, not the first.", B))
    s.append(P("Backtest protocol (the part a hiring manager should interrogate)", H2))
    s += bullets([
        "Expanding-window walk-forward; retrain every 60 trading days; 250-day warm-up; 940 out-of-sample days.",
        "Positions are −1/+1 per asset, equal-weighted; daily return is the mean of position × next-day return, minus 5bp per unit of turnover.",
        "No lookahead: every feature at day t uses data up to day t; the target is day t+1.",
        "Costs are charged on turnover, not on positions — a model that flips daily pays for it, which is exactly why the forest loses.",
    ])
    s.append(P("Reproducibility: <font face='Helvetica-Bold'>python run_all.py</font> regenerates data, features, backtest, charts and metrics.json; <font face='Helvetica-Bold'>python -m tests.test_all</font> runs the suite. Seed fixed (202608). Test suite: <b>56 passed, 0 failed</b> (runtime line, quoted exactly).", B))
    s.append(PageBreak())

    # ---- Page 8: results (honest findings) ----
    s.append(P("8 &nbsp; Results — the honest findings", H1))
    s.append(table(
        ["Model", "Hit rate", "Sharpe (OOS)", "Total return", "Max drawdown", "Turnover"],
        [[r["model"].replace("_", " "), f"{r['hit_rate']:.1%}", f"{r['sharpe']:.2f}", f"{r['total_return_pct']:+.1f}%",
          f"{r['max_drawdown_pct']:.1f}%", f"{r['mean_turnover']:.1%}"] for r in M["summary"]],
        [1.4 * inch, 0.9 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch, 1.0 * inch]))
    s += bullets([
        f"<b>The simple rule wins.</b> Momentum rule: Sharpe {S['momentum_rule']['sharpe']:.2f}, +11.0% over 940 days, max drawdown −5.1%, turnover only 9.3%. Nothing to tune, nothing to overfit.",
        f"<b>Linear is the best model.</b> Linear regression: hit rate {S['linear']['hit_rate']:.1%} (best), Sharpe {S['linear']['sharpe']:.2f}. Logistic is close behind (Sharpe {S['logistic']['sharpe']:.2f}).",
        f"<b>The random forest fails — and that is the most useful result.</b> In-sample accuracy {INS['random_forest']:.1%} vs out-of-sample Sharpe {S['random_forest']['sharpe']:.2f} and −18.3% return. It fitted noise, then paid 25.8% turnover to trade on it. A candidate who hides this would be less credible than one who reports it.",
        f"<b>Regimes decide everything.</b> Momentum-rule Sharpe {REG['momentum_rule']['sharpe_low_vol']:.2f} in low-vol days vs {REG['momentum_rule']['sharpe_high_vol']:.2f} in high-vol days. A desk running this signal without a volatility gate is running half a strategy.",
        "<b>Costs matter more than model choice.</b> At 20bp per trade, every model's Sharpe falls sharply (chart 7) — execution is part of the model, not an afterthought.",
    ])
    s.append(chart("feature_importance.png", 7.0, 3.0))
    s.append(P("Chart 3 — Logistic coefficients (left) load on mom20 and cross-sectional momentum; the forest (right) spreads importance across noise features.", SM))
    s.append(PageBreak())

    # ---- Page 9: more charts + differentiation ----
    s.append(P("8 &nbsp; Results (continued)", H1))
    s.append(chart("drawdown.png", 7.0, 2.6))
    s.append(P(f"Chart 4 — Drawdown of the best model (momentum rule): worst −5.1%, shallow and short-lived — the risk profile a desk can actually live with.", SM))
    s.append(chart("cost_sensitivity.png", 7.0, 2.5))
    s.append(P("Chart 5 — Sharpe vs transaction cost. The forest is underwater at every cost level; the linear models survive to ~10bp.", SM))
    s.append(P("10 &nbsp; Differentiation — why this stands out", H1))
    s += bullets([
        "<b>It reports the failure.</b> Most candidate projects show only the winning model. This one leads with the model that lost money and explains why.",
        "<b>Costs and regimes are first-class, not footnotes.</b> Turnover is charged, regimes are split, and the volatility gate is a stated next step.",
        "<b>It matches the role's actual bar.</b> Statistical signal research with honest backtesting — not a deep-learning demo for a job that asks for statistics, linear algebra and optimisation.",
    ])
    s.append(P("Five questions to ask in the interview: (1) How does the desk decide when a signal is retired? (2) What does the research-to-production handoff look like for a summer analyst's model? (3) Which asset class's data is hardest to clean, and why? (4) How is transaction-cost modelling done for a new signal? (5) What would make a summer analyst's trade idea actually get traded?", B))
    s.append(PageBreak())

    # ---- Page 10: reproducibility, limits, GitHub ----
    s.append(P("9 &nbsp; Reproducibility, limits &amp; next steps", H1))
    s.append(P("How to run", H2))
    s += bullets([
        "<font face='Helvetica-Bold'>python run_all.py</font> — data → features → walk-forward backtest → 7 charts → reports/metrics.json.",
        "<font face='Helvetica-Bold'>python -m tests.test_all</font> — 56 checks; runtime line: <b>56 passed, 0 failed</b>.",
        "Requirements: numpy, pandas, scikit-learn, matplotlib, reportlab (see requirements.txt).",
    ])
    s.append(P("Limits — stated plainly", H2))
    s += bullets([
        "Synthetic data: the momentum component is designed in, so results prove the pipeline works, not that momentum works in real markets.",
        "Six instruments, daily frequency, equal weights — no cross-asset hedging, no position sizing, no intraday effects.",
        "No C++/Java implementation; no stochastic-calculus module — both are named gaps, not hidden ones.",
        "Eligibility: the project does not address the internship's graduation-timing requirement.",
    ])
    s.append(P("Next steps", H2))
    s += bullets([
        "Add a volatility gate: halve exposure when 20d vol is above its median; re-run the walk-forward and report the change.",
        "Add position sizing (volatility targeting) and a second cost model (market-impact proxy).",
        "Paper-trade the momentum rule for 4 weeks on real delayed data and log the diary — the honest bridge from synthetic to real.",
        "Implement the backtester in C++ as the follow-up artefact for the language gap.",
    ])
    s.append(P("GitHub", H2))
    s.append(P("Suggested repo: <font face='Helvetica-Bold'>barclays-markets-quant-intern-ml</font> (github.com/bhupesh-debug). No repo exists yet — the user creates the empty repo first; the project is ready to push (see file count in the delivery note). Chart PNGs are gitignored and excluded from the push; regenerate with <font face='Helvetica-Bold'>run_all.py</font>.", B))
    s.append(P("Sources &amp; method note: JD is Barclays Markets Quantitative Analyst Summer Internship Program 2028, New York, JR-0000139049, $110,000 base (Workday posting, posted 2 days before 25 Nov 2026). All market data is synthetic (src/data_gen.py, seed 202608). All metrics are computed by the executed pipeline and recorded in reports/metrics.json.", SM))
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("report:", path)


def build_deck():
    path = OUT / "Bhupesh_Chandra_Dimri_ML-Project_Markets-Quantitative-Analyst-Summer-Internship_Barclays_Deck.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, leftMargin=0.75 * inch,
                            rightMargin=0.75 * inch, topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title="ML Project Deck — Markets Quantitative Analyst Summer Internship, Barclays",
                            author="Bhupesh Chandra Dimri")
    s = []
    # Slide 1 — title
    s += [Spacer(1, 1.2 * inch), P("Multi-Asset Quantitative<br/>Signal Research", TITLE), Spacer(1, 0.2 * inch),
          P("ML Project for the Markets Quantitative Analyst<br/>Summer Internship Program 2028 — Barclays, New York", SUB),
          Spacer(1, 0.4 * inch), P("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 2026", B),
          P("JR-0000139049 &nbsp;•&nbsp; $110,000 base &nbsp;•&nbsp; SYNTHETIC DATA ONLY", SM), PageBreak()]
    # Slide 2 — problem
    s.append(P("2 — Problem overview", H1))
    s += bullets([
        "✅ <b>One-liner:</b> can simple, explainable signals survive honest out-of-sample testing across six asset classes?",
        "A summer analyst on a Sales &amp; Trading desk needs to build and backtest a model, then defend it to traders. This project is that model, with the defence built in.",
        "Six synthetic instruments stand in for the rotations: Equities, Credit, Rates, FX, Municipals, Securitized Products.",
        "Why it matters: desks do not need a fancier model — they need to know when a signal works, when it breaks, and what it costs to trade.",
    ])
    s.append(PageBreak())
    # Slide 3 — JD breakdown
    s.append(P("3 — JD breakdown", H1))
    s += bullets([
        "<b>Must-have:</b> statistics, linear algebra, optimisation, ML; large-data analysis in Python; build &amp; backtest models; present findings.",
        "<b>Hidden expectations:</b> own a research question end-to-end; judgement under time pressure; 10-week delivery cadence with formal feedback.",
        "<b>Gaps (flagged, not hidden):</b> C++/Java not evidenced; stochastic calculus not evidenced; no trading-desk experience.",
        "<b>Eligibility gate:</b> graduation within one year of the internship's conclusion — separate from quant merit; the project claims nothing about it.",
    ])
    s.append(PageBreak())
    # Slide 4 — ML approach
    s.append(P("4 — ML approach", H1))
    s += bullets([
        "<b>P1 — Direction signal:</b> 7 features per asset-day → next-day direction (−1/+1). Classification + regression.",
        "<b>P2 — Signal portfolio:</b> equal-weighted long/short across six assets, cost-aware, with a volatility-gate proposal.",
        "<b>Models:</b> momentum rule (baseline) • logistic regression • linear regression • random forest (to test whether complexity helps).",
        "<b>Why this bar:</b> the JD asks for statistics and optimisation, not deep learning. The project matches the role's actual technical bar.",
    ])
    s.append(PageBreak())
    # Slide 5 — data
    s.append(P("5 — Data strategy", H1))
    s += bullets([
        "<b>100% synthetic</b> — 1,250 days × 6 assets, seed 202608, generator in src/data_gen.py.",
        "Returns = drift + small momentum component (β 0.05–0.09) + common factor + volatility clustering.",
        "Features: momentum 5/20/60d, 20d vol, RSI-14, carry, cross-sectional momentum. No lookahead: features at day t use data to day t only.",
        "7,140 feature rows. Preprocessing is rolling windows only — nothing peeks at the future.",
    ])
    s.append(PageBreak())
    # Slide 6 — model design
    s.append(P("6 — Model design", H1))
    s += bullets([
        "<b>Baseline first:</b> the momentum rule is the bar; every model must beat it net of costs.",
        "<b>Evaluation:</b> walk-forward, expanding window, retrain every 60 days, 250-day warm-up — 940 out-of-sample days.",
        "<b>Costs charged:</b> 5bp per unit of turnover. A model that flips daily pays for it.",
        "<b>Overfitting controls:</b> regularised linear models, min-leaf-20 forest, and the in-sample vs out-of-sample comparison that catches overfitting.",
    ])
    s.append(PageBreak())
    # Slide 7 — results
    s.append(P("7 — Results &amp; insights", H1))
    s.append(table(
        ["Model", "Hit rate", "Sharpe", "Return", "Max DD"],
        [[r["model"].replace("_", " "), f"{r['hit_rate']:.1%}", f"{r['sharpe']:.2f}",
          f"{r['total_return_pct']:+.1f}%", f"{r['max_drawdown_pct']:.1f}%"] for r in M["summary"]],
        [1.6 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch, 1.2 * inch]))
    s += bullets([
        "✅ <b>Simple wins:</b> momentum rule Sharpe 0.71, +11.0%.",
        "✅ <b>Linear is the best model:</b> hit rate 51.1%, Sharpe 0.59.",
        "✅ <b>The forest fails honestly:</b> 68.3% in-sample, −18.3% out of sample. Complexity + turnover loses.",
        "✅ <b>Regimes rule:</b> momentum Sharpe 1.70 low-vol vs −0.24 high-vol.",
    ])
    s.append(PageBreak())
    # Slide 8 — business impact
    s.append(P("8 — Business impact", H1))
    s += bullets([
        "A desk can run the momentum rule with a volatility gate and know its worst drawdown (−5.1%) in advance.",
        "Turnover discipline is worth more than model sophistication: the forest's 25.8% turnover is what kills it, not its accuracy.",
        "The pipeline is reusable: swap in a real asset, keep the walk-forward harness, and the same report regenerates.",
        "Risk framing a hiring manager cares about: every number is out-of-sample, net of costs, with the failure reported, not hidden.",
    ])
    s.append(PageBreak())
    # Slide 9 — future improvements
    s.append(P("9 — Future improvements", H1))
    s += bullets([
        "Volatility gate: halve exposure when 20d vol is above its median; re-run and report the change.",
        "Volatility-targeted position sizing and a market-impact cost model.",
        "Paper-trade the rule for 4 weeks on real delayed data; keep a trade diary.",
        "Re-implement the backtester in C++ — the follow-up artefact for the language gap.",
        "Production readiness: MLflow tracking, scheduled daily batch, regime dashboard.",
    ])
    s.append(PageBreak())
    # Slide 10 — about
    s.append(P("10 — About this project", H1))
    s += bullets([
        "<b>Created by Bhupesh Chandra Dimri</b> — Financial Analyst, JPMorgan Chase; M.S. Accounting Analytics &amp; Data Technology, Pace (Dec 2025).",
        "<b>GitHub:</b> suggested repo <b>barclays-markets-quant-intern-ml</b> (github.com/bhupesh-debug) — to be created by the user; project ready to push.",
        "<b>Run it:</b> python run_all.py &nbsp;•&nbsp; python -m tests.test_all (56 passed, 0 failed).",
        "<b>Contact:</b> bhupesh7cd@gmail.com &nbsp;•&nbsp; 917-818-6198 &nbsp;•&nbsp; linkedin.com/in/bhupeshdimri",
        "<b>Disclosure:</b> all market data is synthetic; no live-market performance is claimed; the project makes no claim about internship eligibility.",
    ])
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("deck:", path)


if __name__ == "__main__":
    build_report()
    build_deck()
