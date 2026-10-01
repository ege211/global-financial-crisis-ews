# A Machine-Learning-Based Early Warning Framework for Systemic Banking Crises

**Author:** Ege Can  
**Academic Level:** Independent High-School Student Research Project  
**School:** FMV Özel Ispartakule Işık High School, Istanbul (Class of 2027)  
**Date:** September 2026  
**Repository:** `https://github.com/ege211/global-financial-crisis-ews`  
**Evaluation Scope:** 76 World Bank Economies, 1990–2025 Panel, 684 Pooled Out-of-Sample Observations ($T \in [2000, 2008]$)  
**Primary Research Specification:** 6-Variable Extended Logistic Regression (Regime-Qualified)  

---

## Author's Note: What Led Me to Build This and What the Data Actually Taught Me

### 1. The Spark: Why I Didn't Trust Online Tutorials
I got interested in this after reading about the 2008 crash. What confused me was simple: if this crisis shook the entire world economy, why did standard forecasting tools completely miss it? When I looked up tutorials and GitHub repos about "predicting financial crises with machine learning," almost all of them claimed 95% to 99% accuracy. That immediately felt off to me. Predicting a rare banking crisis isn't like classifying pictures of cats and dogs. When I inspected their code, I saw they were basically letting the model peek into the future—shuffling years randomly and training on 2009 data to "predict" 2007. I wanted to build an end-to-end framework from scratch to see what happens when the computer is strictly forbidden from cheating.

### 2. The 97% Trap (Finding a Needle in a Haystack)
The first shock came when I assembled the actual ground-truth data from the World Bank and the IMF. Across 76 countries evaluated over nine years (684 country-years), there were only 19 actual crisis onsets. That’s less than 3%. When I ran my very first basic model, my screen showed 97.2% accuracy. For about ten seconds, I thought I was a genius. Then I looked at the confusion matrix: the model was simply guessing "no crisis" for every single country, every single year. It got an "A" on paper while being 100% useless in the real world. That was my first big lesson: standard accuracy is a complete trap when you are looking for rare disasters. I had to throw it out and focus entirely on precision, recall, and the real cost of missing a crisis.

### 3. The "No Time-Travel" Rule
Stopping the computer from cheating across time was the hardest engineering challenge in the project. In regular school coding projects, you just shuffle the data and split it 80/20. But in economics, that’s time travel. You can’t use what happened in Spain in 2008 to guess what happened in 2004. I had to build an expanding-window pipeline where the model only learns from past years to forecast the single next year. Even basic steps like filling in missing numbers or scaling features had to be calculated strictly on past data before touching the test year. It took weeks of debugging and 80 automated unit tests to build that wall, but it ensured every single test prediction was honest.

### 4. What the Data Actually Told Me (The 2007 Reality Check)
When I added private debt and current account deficits into the model, the pooled numbers looked like a massive win at first. The detection rate jumped from 21% to 47.37%, catching 9 out of 19 crises. But when I audited the results year by year, I got a sobering reality check. Out of the 19 crises in the test period, 15 happened in one single year: right before the 2008 crash. In that specific wave (countries like Spain, Ireland, and Iceland that borrowed way too much), my model did well—catching 9 out of 15. But across all the other eight years combined, there were 4 other crises (like the UK or Uruguay), and my model caught exactly zero of them. Outside 2007, its performance was literally worse than flipping a coin (ROC-AUC = 0.2301).

### 5. My Main Takeaway
At first, seeing the model fail outside 2007 felt disappointing. But digging into the history showed me why: my model was specifically tracking domestic debt bubbles and trade deficits. It had no way of seeing bank runs like Northern Rock in the UK or money fleeing Uruguay. That taught me what data science in economics actually is. An early warning model is not an all-knowing crystal ball; it is more like a smoke detector calibrated to smell one specific type of fire. Admitting what my model couldn't do wasn't a failure—it was the most honest and valuable finding of the entire project.

---

## Abstract

Systemic banking crises cause devastating economic harm, yet predicting them using quantitative data remains a formidable challenge. In this independent research project, I investigate whether standard macro-financial indicators can provide practical early warning signals for banking crises without falling into common statistical traps. Using an annual panel of 76 economies spanning 1990–2025 constructed from World Bank Development Indicators and the IMF Systemic Banking Crises Database (Laeven & Valencia, 2026), I evaluate linear logistic regression models against random forest tree ensembles across a baseline set of 4 economic flow variables and an extended set of 6 balance-sheet indicators. To keep the testing realistic and prevent chronological cheating ("lookahead leakage"), I test models across nine sequential test years (2000–2008) using expanding training windows.

The 6-variable Extended Logistic Regression achieved the highest overall detection rate among the models I tested, increasing crisis recall from 21.05% ($4/19$) to 47.37% ($9/19$), while generating 187 false alarms (Precision = 4.59%). In contrast, Random Forest models struggled because banking crises are so rare (only 2.78% of the data); the tree models tended to play it safe and guess "no crisis", missing 15 to 17 of the 19 crisis events.

Crucially, when I audited the results year by year, I uncovered an important reality check: the predictive advantage of the model was concentrated almost entirely in the single year right before the 2008 Global Financial Crisis (2007), where 15 of the 19 test crises occurred. In that single 2007 wave, the model caught 9 out of 15 crises by detecting heavy domestic private debt (median 142.2% of GDP) and wide current account deficits (median -7.29% of GDP). But outside 2007, across the four other crises in the test period, the model detected zero crises (0.0% Recall). This demonstrates that early warning models are not universal crystal balls; rather, this model functioned as a specific detector for credit-driven housing and trade bubbles like 2008, rather than an all-purpose crisis predictor.

---

## 1. Introduction

Systemic banking crises are among the most destructive events in modern economic history. When banks fail or face runs on their deposits, businesses cannot borrow, unemployment spikes, and governments often take on massive debts to bail out the financial system. Following the 2008 Global Financial Crisis (GFC), economists and policymakers around the world redoubled their efforts to build quantitative "Early Warning Systems" (EWS) to detect financial stress before it turns into a full collapse.

As a high school student studying economics and statistics, I wanted to see if these models actually work when tested realistically. Building an early warning model sounds straightforward, but digging into the literature reveals three major hurdles:
1. **The 97% Accuracy Trap (Rare Events):** Banking crises are statistically rare. Across 76 countries over 9 test years, only 19 crises occurred out of 684 country-years (just 2.78% of the time). Under such extreme imbalance, a naive model that simply guesses "no crisis" every single year achieves a deceptively high 97.2% accuracy while being completely useless.
2. **Crises Change Over Time (Regime Shifts):** Financial crises do not all look the same. An indicator that worked well for Asian currency crises in 1997 might completely miss a banking panic driven by complex mortgage bonds in Europe or the US in 2008.
3. **Cheating the Clock (Look-Ahead Leakage):** Many machine-learning projects randomly shuffle data using standard cross-validation. In financial time series, this is fatal: it allows a model predicting the past to train on future information that wasn't known at the time.

In this project, I address these challenges by building an end-to-end, leak-free evaluation pipeline in Python. Instead of using complex algorithms as mysterious black boxes, I compare a transparent linear model (Logistic Regression) against a decision-tree model (Random Forest). I test whether adding banking debt and trade deficits to basic GDP and inflation numbers actually improves the ability to spot crises, using an expanding historical window so the model never peeks into the future.

My findings offer both an exciting result and an honest cautionary tale: adding private credit and trade deficit data doubled the model's crisis detection rate from 21% to 47%. But when I audited the results by year, I discovered that almost all of that success came from the 2007 pre-GFC wave. Outside 2007, the model caught zero crises. Furthermore, the Random Forest model performed poorly because the extreme rarity of crises caused it to predict near-zero risk everywhere. This taught me that machine learning in economics requires deep humility: models reflect the specific historical episodes they were trained on, not universal laws.

---

## 2. Research Questions and Hypotheses

The central research question guiding my project is:

> **"Can standard macro-financial indicators provide reliable early warning signals for systemic banking crises?"**

To explore this question systematically, I divided it into two student sub-questions:
1. **The Role of Debt and Trade Imbalances:** Does adding banking debt (`private_credit_pct_gdp`) and external trade deficits (`current_account_pct_gdp`) to basic economic indicators (GDP growth, inflation, and foreign exchange reserves) improve the ability to detect upcoming crises?
2. **Historical Consistency:** Does the model work consistently across different years, or are its successes concentrated in a single historic crisis wave (such as 2007–2008)?

### Hypotheses

Based on macroeconomic theory and economic history:
- **Hypothesis 1 (Balance-Sheet Hypothesis):** Because modern banking crises often stem from debt bubbles rather than just slowing GDP growth, adding private credit leverage and trade deficits will improve out-of-sample crisis detection compared to a flow-only baseline.
- **Hypothesis 2 (Regime Consistency Hypothesis):** If domestic credit expansion is a universal warning sign, the extended model should catch crises across multiple different years, rather than just in the 2007 Global Financial Crisis wave.

---

## 3. Literature and Conceptual Motivation

The econometric literature on crisis early warning systems traces its origins to the seminal balance-of-payments and currency crisis signaling models pioneered by Kaminsky, Lizondo, and Reinhart (1998) and Kaminsky and Reinhart (1999). In their classic "Twin Crises" framework, Kaminsky and Reinhart demonstrated that banking sector distress and currency collapses frequently reinforce one another. Banking crises typically emerge following financial liberalization and rapid domestic credit expansions, which subsequently leave the financial system vulnerable to capital flight, reserve exhaustion, and currency depreciations.

In the decades following the 1997 Asian Financial Crisis and the 2008 Global Financial Crisis, macroeconomic research increasingly converged on the central role of the credit cycle. Schularick and Taylor (2012), examining nearly 140 years of historical data across 14 advanced economies, established that "credit growth is the single best predictor of financial instability." They showed that the post-World War II era witnessed a massive decoupling of money aggregates from credit aggregates—an era they termed the "Second Financialization"—during which bank balance sheets expanded dramatically through private credit creation rather than deposit accumulation. The broader historical literature on debt cycles and financial crises, synthesized by Reinhart and Rogoff (2009), emphasizes that systemic banking crises are protracted balance-sheet phenomena preceded by sustained run-ups in public and private indebtedness. In related work, Gourinchas and Obstfeld (2012) conducted a systematic empirical study of financial crises in advanced and emerging market economies from 1973 to 2010, confirming that domestic credit expansion and real currency appreciation are the most robust and consistent predictors of both banking and currency crises across all economic developmental tiers.

From a conceptual and economic standpoint, the six predictors evaluated in this study represent three distinct, well-theorized transmission channels of macro-financial vulnerability:

### 1. Macroeconomic Deterioration and Monetary Instability
- **Real GDP Growth (`gdp_growth`):** Pre-crisis decelerations in real economic activity weaken corporate profitability, increase debt-servicing burdens on domestic borrowers, and lead to an accumulation of non-performing loans (NPLs) across commercial banks. Conversely, unsustainably high GDP growth during boom phases can reflect overheating and asset price bubbles.
- **Consumer Price Inflation (`inflation`):** High or volatile inflation distorts financial contracting, erodes bank capital in real terms, and frequently triggers aggressive monetary tightening cycles by central banks, which can abruptly prick domestic credit and real estate bubbles.

### 2. External Buffers and Balance-of-Payments Pressure
- **Gross International Reserves (`reserves_usd`):** Substantial foreign exchange reserve stocks serve as a liquid national liquidity buffer. In emerging and open economies, high reserve levels bolster central bank credibility, provide insurance against sudden stops in international financing, and enable monetary authorities to supply emergency foreign-exchange liquidity to domestic banks during wholesale liquidity runs.
- **Reserves YoY Growth Rate (`reserves_usd_yoy_pct_change`):** While reserve levels capture steady-state liquidity stocks, sharp annual reserve drawdowns signal acute balance-of-payments distress, capital flight, and speculative pressure on domestic currency pegs, directly presaging systemic banking stress.

### 3. Balance-Sheet Leverage and External Imbalances
- **Domestic Private Credit to GDP (`private_credit_pct_gdp`):** Private credit depth directly captures banking-sector asset expansion and leverage. Rapid credit growth relative to the real economy inflates collateral values, lowers lending standards, and increases banking system exposure to debt defaults when the credit cycle turns.
- **Current Account Balance to GDP (`current_account_pct_gdp`):** Large, persistent current account deficits indicate that domestic investment exceeds domestic savings, requiring continuous inflows of foreign capital (Obstfeld, 2012). When banking systems intermediate these external flows through short-term wholesale foreign borrowing, they become acutely vulnerable to global liquidity freezes and capital reversals.

*Methodological Note on Literature Scope:* While extensive literature exists regarding alternative indicators—such as house-price indices, bank-level equity ratios, and financial condition indices (FCIs)—this study intentionally restricts its candidate predictor set to indicators with broad, objective, historical availability across both advanced and developing economies, adhering to the pre-registered research constraints of Phases 1 through 5. Additional literature on high-frequency interbank network contagion and sovereign spread dynamics is acknowledged as requiring external verification beyond the annual multilateral panel evaluated herein.

---

## 4. Data and Objective Panel Universe

All data analyzed in this research project derive strictly from authoritative, multilateral institutions: the World Bank and the International Monetary Fund (IMF).

### Country Universe
The research universe consists of an objective, balanced panel of **76 World Bank member economies** spanning all major geographic regions and income classifications. The 76 economies were established in Phase 4B following explicit data availability and macroeconomic relevance criteria, requiring continuous annual coverage across the primary macroeconomic series from 1990 through 2025. The universe encompasses 28 advanced economies (e.g., the United States, Germany, Japan, United Kingdom, France) and 48 emerging and developing economies (e.g., Brazil, China, India, Mexico, Turkey, Nigeria, Indonesia).

### Observation Period and Evaluation Slices
The historical dataset spans **1990 to 2025** (36 calendar years per country), comprising a total panel size of 2,736 country-year observations. Within this panel, the temporal validation design partitions the data into:
- **Pre-Validation Burn-in / Training Initialization Period (1990–1999):** 10 calendar years ($760$ country-year observations) utilized exclusively for initial model estimation.
- **Out-of-Sample Evaluation Window ($T = 2000, \dots, 2008$):** 9 sequential evaluation test folds ($76 \times 9 = 684$ evaluated country-year observations), during which out-of-sample predictions are strictly tested.
- **Post-Evaluation Observation Period (2009–2025):** 17 calendar years ($1,292$ country-year observations) reserved for future prospective validation.

### Candidate Predictor Indicators
The candidate predictors were extracted from the World Bank World Development Indicators (WDI) API using versioned request templates and archived JSON payloads (`data/raw/full_research/`). The six indicators analyzed across the Baseline and Extended models are summarized in Table 1.

| Variable Name | World Bank Indicator Code | Description | Units | Economic Role |
| :--- | :--- | :--- | :--- | :--- |
| `gdp_growth` | `NY.GDP.MKTP.KD.ZG` | Real GDP Growth | Annual % | Macroeconomic flow; output momentum |
| `inflation` | `FP.CPI.TOTL.ZG` | CPI Inflation Rate | Annual % | Monetary instability; price overheating |
| `reserves_usd` | `FI.RES.TOTL.CD` | Gross International Reserves | Current USD | National foreign-exchange liquidity stock |
| `reserves_usd_yoy_pct_change` | Derived from `FI.RES.TOTL.CD` | Annual Reserves Growth Rate | Annual % | Balance-of-payments flow pressure |
| `private_credit_pct_gdp` | `FS.AST.PRVT.GD.ZS` | Domestic Credit to Private Sector | % of GDP | Banking sector leverage; credit depth |
| `current_account_pct_gdp` | `BN.CAB.XOKA.GD.ZS` | Current Account Balance | % of GDP | External macro-financial balance |

*Table 1: Data dictionary and indicator sources for the 6-variable research specification.*

---

## 5. Crisis Label Construction

Establishing an objective, uncorrupted ground truth for systemic banking crises is critical for early warning research. Subjective crisis dates or ad-hoc post-event classifications introduce severe survivorship and confirmation biases.

### Authoritative Crisis Source
Ground-truth crisis labels derive strictly from the official International Monetary Fund (IMF) Systemic Banking Crises Database published by Luc Laeven and Fabian Valencia (IMF Working Paper WP/26/94, *Systemic Banking Crises Database: A 2026 Update*, building on Laeven & Valencia, 2013, 2018, 2020). The Laeven & Valencia chronology is widely recognized in academic economics as the definitive historical record of systemic banking crises.

Under the IMF definition, an episode is classified as a systemic banking crisis if it satisfies two rigorous conditions:
1. **Significant Signs of Financial Distress:** Significant financial distress in the banking system, demonstrated by massive bank runs, substantial non-performing loan spikes, capital erosions, or widespread bank liquidations.
2. **Significant Banking Policy Intervention Measures:** Significant public policy interventions in response to banking distress, defined as at least three of the following six policy actions:
   - Extensive emergency liquidity assistance (liquidity support exceeds 5% of deposits and liabilities to nonresidents);
   - Substantial bank restructuring costs (gross fiscal restructuring costs exceed 3% of GDP);
   - Significant bank nationalizations;
   - Significant guarantees on bank liabilities;
   - Large-scale asset purchases (government purchases of bank assets exceed 5% of GDP);
   - Severe deposit freezes or bank holidays.

### Exclusion of Borderline Episodes and Continuation Years
To avoid noise from minor financial distress episodes, I included only **non-borderline** systemic banking crises as designated by the IMF source. Furthermore, my target is strictly defined as the **initial crisis onset year** ($t_{\text{onset}}$). Subsequent crisis continuation years (years during which the crisis was ongoing but did not start) are coded as calm observations in forward-looking early warning pipelines, preventing post-crisis economic fallout from contaminating the pre-crisis predictive signal.

### The $t \to t+1$ Forward Prediction Target
The primary predictive target is a binary indicator $y_{c,t}$ defined for country $c$ in calendar year $t$:

$$y_{c,t} = \begin{cases} 1 & \text{if a systemic banking crisis onset occurs in country } c \text{ at calendar year } t+1 \\ 0 & \text{otherwise} \end{cases}$$

This structure enforces a strict one-year-ahead early warning horizon ($t \to t+1$). All predictor values in vector $\mathbf{x}_{c,t}$ are measured at calendar year $t$. Therefore, when predicting a crisis onset occurring in 2008, the model conditions strictly on macroeconomic indicators observed in 2007. Information from the crisis year $t+1$ (such as the collapse of Lehman Brothers in September 2008 or emergency policy rate cuts in late 2008) is strictly excluded from year $t$ features, eliminating contemporaneous contamination.

### Crisis Counts Across Panel Slices
The distinction between panel-wide labels, out-of-sample evaluation events, and calendar onset years is vital for statistical precision:
- **Full Panel Labels (1990–2025):** Across all 76 countries and 36 years ($N = 2,736$), the dataset contains exactly **44 positive forward crisis onset labels** ($y_{c,t} = 1$).
- **Out-of-Sample Evaluation Events ($T \in [2000, 2008]$):** Across the 9 test folds comprising 684 evaluated country-years, there are exactly **19 positive forward crisis onset events** (an evaluation base rate of $19 / 684 = 2.78\%$).
- **Temporal Distribution:** Of the 19 evaluation events, **15 events cluster in the single test fold $T = 2007$** (predicting 2008 GFC onsets). The remaining **4 events are distributed across the other 8 evaluation folds** (Uruguay evaluated in 2001, Dominican Republic in 2002, United Kingdom in 2006, and Nigeria in 2008).

---

## 6. Feature Engineering and Preprocessing

To prevent data leakage, all preprocessing transformations, imputation parameters, and scaling statistics must be computed strictly on historical training data.

### Reserve Growth Derivation and Lagged Missingness
The annual percentage growth rate of international reserves is computed as:

$$\Delta R_{c,t} = \left( \frac{\text{reserves\_usd}_{c,t} - \text{reserves\_usd}_{c,t-1}}{\text{reserves\_usd}_{c,t-1}} \right) \times 100$$

Because computing $\Delta R_{c,t}$ requires a one-year lag, observations for the initial panel year (1990) lack a lagged value. This represents a structural missingness of exactly 76 observations (2.78% of the panel) in 1990.

### Leak-Free Train-Only Preprocessing Protocol
Standard machine-learning workflows frequently fit imputers or scalers across the complete dataset prior to cross-validation. In temporal panel analysis, this constitutes severe future data leakage:
- If median imputation uses the whole sample, the median value of private credit in 2002 is influenced by the massive credit expansion of 2006–2007.
- If z-score normalization standardizes features using whole-sample means and variances, information regarding future crisis volatility leaks into historical training folds.

To eliminate future leakage, I enforced a strict **train-only preprocessing protocol** for each sequential expanding test fold $T \in [2000, 2008]$:
1. **Training Partition Isolation:** For evaluation test year $T$, the training sample is restricted strictly to historical years $t \in [1990, T-1]$. Test fold $T$ is completely quarantined.
2. **Fold-Fitted Median Imputation:** Missing values in training fold predictors are imputed using the median of that specific feature computed strictly across the training partition:
   $$\hat{x}_{j}^{\text{imp}} = \text{Median}\left(\{ x_{i,j} \}_{i \in \text{Train}_T}\right)$$
   The exact training-derived median $\hat{x}_{j}^{\text{imp}}$ is then applied forward to impute any missing values in test fold $T$.
3. **Fold-Fitted Standardization for Logistic Models:** Because Logistic Regression coefficients depend on predictor scales and regularization penalties, training features are standardized to zero mean and unit variance:
   $$z_{i,j} = \frac{x_{i,j} - \mu_{j,\text{Train}}}{\sigma_{j,\text{Train}}}$$
   The parameters $\mu_{j,\text{Train}}$ and $\sigma_{j,\text{Train}}$ are fitted strictly on the training partition and applied forward to test year $T$.
4. **Tree Model Invariance:** Random Forest ensembles do not require z-score scaling, as decision-tree split criteria are invariant to monotonic scale transformations. Imputations for tree models use identical fold-fitted training medians.

---

## 7. Modeling Framework

I evaluated four model setups: two model classes (Logistic Regression vs. Random Forest) across two feature specifications (Baseline vs. Extended).

### Model Architecture Hierarchy
1. **Baseline Logistic Regression (4 variables):** Linear logit specification conditioning on macroeconomic flows (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`).
2. **Extended Logistic Regression (6 variables):** Linear logit specification augmenting the baseline with banking leverage (`private_credit_pct_gdp`) and external imbalances (`current_account_pct_gdp`).
3. **Baseline Random Forest (4 variables):** Non-linear ensemble conditioning on the 4 flow variables.
4. **Extended Random Forest (6 variables):** Non-linear ensemble conditioning on the 6 balance-sheet and flow variables.

### Logistic Regression Formulation
The Logistic Regression model specifies the conditional probability of a crisis onset in country $c$ at year $t+1$, given the standardized predictor vector $\mathbf{z}_{c,t} \in \mathbb{R}^K$, via the standard sigmoid link function:

$$P(y_{c,t} = 1 \mid \mathbf{z}_{c,t}) = \sigma(\beta_0 + \mathbf{\beta}^T \mathbf{z}_{c,t}) = \frac{1}{1 + \exp\left(-(\beta_0 + \sum_{k=1}^K \beta_k z_{c,t,k})\right)}$$

The model is estimated by minimizing the cost-sensitive weighted negative log-likelihood (cross-entropy) loss function:

$$\mathcal{L}(\mathbf{\beta}) = - \sum_{i \in \text{Train}} \left[ w_1 y_i \log(\hat{p}_i) + w_0 (1 - y_i) \log(1 - \hat{p}_i) \right]$$

where weights $w_1$ and $w_0$ implement balanced class weighting inversely proportional to training class frequencies:

$$w_1 = \frac{N_{\text{train}}}{2 \cdot N_{\text{crisis}}}, \quad w_0 = \frac{N_{\text{train}}}{2 \cdot (N_{\text{train}} - N_{\text{crisis}})}$$

Logistic Regression serves as my **Primary Research Model** because it is transparent and easy to interpret. Every parameter $\beta_k$ translates directly into an odds ratio $\text{OR}_k = \exp(\beta_k)$, guaranteeing monotonic risk behavior and avoiding unconstrained split overfitting in sparse data regimes.

### Random Forest Ensemble Formulation
The Random Forest model serves as a **Nonlinear Benchmark** to test whether decision trees can detect complex relationships that a linear model might miss. Estimators across both model classes are implemented using the `scikit-learn` framework (Pedregosa et al., 2011). Each Random Forest ensemble comprises $B = 100$ balanced bootstrap trees with maximum tree depth constrained to $d = 4$ to prevent overfitting in this rare-crisis setting. Each tree is trained on a bootstrap sample with balanced class reweighting, and individual tree predictions are averaged to yield the ensemble probability:

$$\hat{P}_{\text{RF}}(y_{c,t} = 1 \mid \mathbf{x}_{c,t}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x}_{c,t})$$

### Pre-Registration and No Test-Set Optimization
To ensure scientific integrity, **zero hyperparameter tuning or feature selection was performed on the out-of-sample evaluation window**. All model architectures, tree depths ($d=4$), ensemble sizes ($B=100$), predictor candidate lists, and decision thresholds ($\tau = 0.50$) were locked prior to evaluating test folds $T \in [2000, 2008]$. Model metrics reflect genuine out-of-sample performance rather than post-hoc optimization.

---

## 8. Validation Design

The validation methodology is the methodological cornerstone of this investigation. In time-series panel forecasting, the design of the validation split determines whether reported results represent genuine predictive power or spurious retrospective correlation.

### Why Random Train/Test Splits Are Fatally Flawed
In standard machine-learning benchmarks (e.g., computer vision or tabular cross-sectional data), $k$-fold cross-validation or randomized 80/20 train/test splits are standard practice. However, applying randomized cross-validation to macroeconomic panel crisis forecasting violates the basic temporal ordering of data and produces severe methodological failures:
1. **Future Look-Ahead Contamination:** If the 2008 crisis observation for Spain is placed in the training set while the 2006 calm observation for Spain is placed in the test set, the model trains on future crisis aftermath data to predict past calm periods.
2. **Contemporaneous Cross-Country Leakage:** If the 2008 crisis for France is in the training set and the 2008 crisis for Germany is in the test set, the model conditions on contemporaneous global shock waves to "predict" an event occurring at the exact same moment in calendar time.
3. **Artificial Performance Inflation:** Randomized cross-validation in crisis forecasting consistently yields inflated ROC-AUC scores (> 0.85) that vanish completely when the models are evaluated in true prospective walk-forward settings.

### Expanding-Window Out-of-Sample Protocol
To guarantee strict temporal realism and zero future leakage, I implemented an **expanding-window out-of-sample temporal validation design** (illustrated in Figure 1).

```
Training Horizon (Expands by +1 Year)           Test Year T (Fixed Horizon)
Fold 1 (T=2000): [1990 —————————— 1999] (10 yrs)  --> [2000] (76 countries, out-of-sample)
Fold 2 (T=2001): [1990 ————————————— 2000] (11 yrs) --> [2001] (76 countries, out-of-sample)
Fold 3 (T=2002): [1990 ———————————————— 2001] (12 yrs) --> [2002] (76 countries, out-of-sample)
Fold 4 (T=2003): [1990 ——————————————————— 2002] (13 yrs) --> [2003] (76 countries, out-of-sample)
Fold 5 (T=2004): [1990 —————————————————————— 2003] (14 yrs) --> [2004] (76 countries, out-of-sample)
Fold 6 (T=2005): [1990 ————————————————————————— 2004] (15 yrs) --> [2005] (76 countries, out-of-sample)
Fold 7 (T=2006): [1990 ———————————————————————————— 2005] (16 yrs) --> [2006] (76 countries, out-of-sample)
Fold 8 (T=2007): [1990 ——————————————————————————————— 2006] (17 yrs) --> [2007] (76 countries, out-of-sample)
Fold 9 (T=2008): [1990 —————————————————————————————————— 2007] (18 yrs) --> [2008] (76 countries, out-of-sample)
```
*Figure 1: Expanding-window out-of-sample evaluation architecture across folds $T=2000, \dots, 2008$.*

The protocol operates under strict mathematical constraints:
- **Evaluation Folds:** Exactly nine sequential folds: $T \in \{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008\}$.
- **Minimum Training Horizon:** 10 continuous years ($1990 \le t \le T-1$). The initial fold ($T=2000$) trains on 1990–1999 ($N_{\text{train}} = 760$); the final fold ($T=2008$) trains on 1990–2007 ($N_{\text{train}} = 1,368$).
- **Strict Out-of-Sample Testing:** In each fold $T$, models generate forward crisis predictions for all 76 countries. Predictions are generated exclusively using the fitted parameters and preprocessing statistics derived from $[1990, T-1]$.
- **Total Evaluated Observations:** Across the 9 folds, exactly $76 \times 9 = 684$ out-of-sample predictions are generated.
- **Total Positive Evaluation Events:** Across the 684 evaluations, exactly 19 forward crisis onset events occur (2.78% prevalence).
- **Locked Decision Threshold:** The classification decision threshold is held constant at $\tau = 0.50$ across all models and folds:
  $$\hat{y}_{c,t} = \mathbb{I}(\hat{P}_{c,t} \ge 0.50)$$

---

## 9. Evaluation Metrics for Rare-Event Imbalance

Evaluating probabilistic early warning models under extreme rare-event class imbalance ($2.78\%$ positive prevalence) requires metrics that directly penalize missed crises and excessive false alarms.

### Why Standard Accuracy Is Uninformative
Overall classification accuracy is defined as:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

In my evaluation panel, where 665 of 684 observations are tranquil ($\text{TN} = 665$) and only 19 are crises ($\text{Pos} = 19$), a trivial classifier predicting $\hat{y} = 0$ for all observations achieves:

$$\text{Accuracy}_{\text{trivial}} = \frac{0 + 665}{684} = 97.22\%$$

Despite high accuracy, this model is completely useless for policy early warning because its crisis recall is exactly 0.0%. Consequently, overall accuracy is rejected as a primary performance metric.

### Primary Metrics
To assess model performance without getting fooled by high overall accuracy, I reported:
1. **Precision-Recall Area Under the Curve (PR-AUC):** PR-AUC integrates precision over all possible recall thresholds. In severely imbalanced panels, PR-AUC is vastly more informative than ROC-AUC because it focuses directly on the minority positive class (Saito & Rehmsmeier, 2015). The baseline for PR-AUC under random guessing equals the unconditional class prevalence ($\pi = 19 / 684 = 0.0278$). Any model achieving PR-AUC $> 0.0278$ provides positive predictive value above the uninformative prior.
2. **Receiver Operating Characteristic Area Under the Curve (ROC-AUC):** Measures the probability that the model ranks a randomly chosen crisis observation higher than a randomly chosen calm observation across all possible thresholds. A score of 0.50 indicates uninformative random ranking; scores $< 0.50$ indicate ranking inversion.
3. **Recall / Sensitivity / True Positive Rate (TPR):** The proportion of actual crises successfully detected:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{\text{TP}}{19}$$
4. **Precision / Positive Predictive Value (PPV):** The proportion of issued alarms that correspond to genuine crises:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
5. **F1-Score:** The harmonic mean of precision and recall:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
6. **Brier Score:** The mean squared error of probabilistic predictions (Brier, 1950):
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
   The Brier score measures calibration and probability accuracy. Lower scores indicate superior probabilistic calibration.

---

## 10. Baseline Model Results

The 4-variable macroeconomic flow baseline models (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`) establish the benchmark against which balance-sheet expansions are evaluated. The empirical results across the 684 pooled out-of-sample evaluations ($N_{\text{crisis}} = 19$) are reported in Table 2.

| Performance Metric | Baseline Logistic Regression (4 vars) | Baseline Random Forest (4 vars) | Benchmark Evaluation |
| :--- | :---: | :---: | :--- |
| **Out-of-Sample Observations ($N$)** | 684 | 684 | Exact match across 9 folds |
| **Positive Crisis Events ($N_{\text{pos}}$)**| 19 | 19 | Unconditional prevalence = 2.78% |
| **PR-AUC** | **0.0278** | **0.0256** | Both hover at the random prior ($\pi = 0.0278$) |
| **ROC-AUC** | **0.5106** | **0.4712** | LR marginally above 0.50; RF slightly inverted |
| **Brier Score** | 0.2512 | **0.1524** | RF achieves lower squared error via base-rate compression |
| **Recall (TPR at $\tau = 0.50$)** | **21.05%** ($4/19$) | **21.05%** ($4/19$) | Both detect exactly 4 of 19 crisis events |
| **Precision (PPV at $\tau = 0.50$)** | 1.89% ($4/212$) | **2.42%** ($4/165$) | Severe false-alarm burden across both models |
| **F1-Score** | 0.0346 | **0.0435** | Moderate due to low precision |
| **True Positives (TP)** | 4 | 4 | Identified crisis events |
| **False Positives (FP)**| 208 | 161 | Non-crisis observations issuing alarms |
| **False Negatives (FN)**| 15 | 15 | Actual crises missed by the model |
| **True Negatives (TN)** | 457 | 504 | Correctly identified tranquil observations |

*Table 2: Performance metrics for Baseline models across 684 pooled out-of-sample evaluations ($T=2000,\dots,2008$).*

### Cautious Interpretation of Baseline Performance
The baseline results demonstrate the inherent difficulty of macroeconomic crisis early warning:
1. **Uninformative Discrimination:** The Baseline Logistic Regression achieves a PR-AUC of $0.0278$, which is virtually identical to the unconditional sample prevalence of the positive class ($19 / 684 = 0.02778$). Its ROC-AUC of $0.5106$ indicates that ranking discrimination is barely superior to a random coin toss.
2. **The Non-Linear Random Forest Inversion:** The Baseline Random Forest fails to outperform linear logistic regression, achieving a lower PR-AUC ($0.0256$) and an inverted ROC-AUC ($0.4712 < 0.50$). Although Random Forest produces fewer false alarms (161 vs. 208) and a substantially lower Brier score ($0.1524$ vs. $0.2512$), this reflects the model simply guessing low risk everywhere rather than genuinely spotting crises. Both models detect only 4 of the 19 crises, missing 78.9% of historical events.

---

## 11. Extended Model Results

Augmenting the 4-variable flow baseline with `private_credit_pct_gdp` and `current_account_pct_gdp` yields the pre-registered 6-variable Extended Model. Table 3 presents the comparative performance across all four evaluated architectures.

| Metric | Baseline LR (4 vars) | Extended LR (6 vars) | Baseline RF (4 vars) | Extended RF (6 vars) | Primary Comparison |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Predictor Count** | 4 | **6** | 4 | 6 | +2 balance-sheet features |
| **PR-AUC** | 0.0278 | **0.0344** | 0.0256 | 0.0274 | Extended LR is highest (+23.7% over baseline) |
| **ROC-AUC** | 0.5106 | **0.5734** | 0.4712 | 0.4715 | Extended LR is highest (+0.0628 gain) |
| **Brier Score** | 0.2512 | 0.2482 | 0.1524 | **0.1347** | RF lower due to base-rate probability shrinkage |
| **Recall ($\tau=0.50$)** | 21.05% ($4/19$) | **47.37%** ($9/19$) | 21.05% ($4/19$) | 10.53% ($2/19$) | Extended LR detects +5 additional crises (+125%) |
| **Precision ($\tau=0.50$)**| 1.89% ($4/212$) | **4.59%** ($9/196$) | 2.42% ($4/165$) | 1.61% ($2/124$) | Extended LR achieves highest precision |
| **F1-Score** | 0.0346 | **0.0837** | 0.0435 | 0.0280 | Extended LR more than doubles F1-score |
| **True Positives (TP)** | 4 | **9** | 4 | 2 | Crises caught at threshold 0.50 |
| **False Positives (FP)**| 208 | 187 | 161 | **122** | Extended LR issues -21 fewer false alarms |
| **False Negatives (FN)**| 15 | **10** | 15 | 17 | Extended LR misses the fewest crises |
| **True Negatives (TN)** | 457 | 478 | 504 | **543** | Correctly classified tranquil observations |

*Table 3: Consolidated out-of-sample performance comparison across all four evaluated architectures ($N=684, \text{Pos}=19$).*

### Empirical Takeaways from Model Comparison
1. **Extended Logistic Regression Dominates on Pooled Discrimination:** The 6-variable Extended Logistic Regression achieves the highest pooled PR-AUC ($0.0344$), highest ROC-AUC ($0.5734$), highest Recall ($47.37\%$), highest Precision ($4.59\%$), and highest F1-score ($0.0837$) among all four tested specifications. It successfully identifies 9 out-of-sample crises while simultaneously reducing false alarms from 208 to 187.
2. **Failure of Non-Linear Ensembles to Improve Detection:** Contrary to common expectations that machine learning ensembles inherently outperform linear models, Random Forest ensembles performed poorly in rare-event detection. The Extended Random Forest detected only 2 of 19 crises (Recall = 10.53%), missing 17 crisis events. Its ROC-AUC ($0.4715$) remained inverted below 0.50.
3. **The Probability Compression Trade-off:** While Random Forest achieves an attractive Brier score ($0.1347$ vs. $0.2482$ for Extended LR), this happens because it simply plays it safe. In a dataset where 97.2% of years are peaceful, a model that guesses near-zero risk for everyone gets a low mathematical error penalty, but it completely misses real crises by never daring to predict a probability above 50%.

---

## 12. Robustness and Sensitivity Analysis

To examine the stability of these empirical findings, Phase 5C subjected the models to extensive robustness evaluations across four pre-registered dimensions. The results are summarized in Table 4.

| Robustness Dimension | Baseline Condition | Perturbation Condition | Empirical Finding | Methodological Implication |
| :--- | :--- | :--- | :--- | :--- |
| **1. Inflation Winsorization** | Raw CPI inflation (untrimmed) | Winsorized at [1st, 99th] percentiles across training folds | Logistic PR-AUC drops from 0.0278 to 0.0219 (-21.2%); Recall drops to 5.26% (1 detection). RF unchanged. | Linear logit models rely heavily on extreme inflation tails for crisis separation. Truncating outliers destroys predictive signal. |
| **2. Class Weight Sensitivity** | Balanced loss weighting ($w_1 \propto 1/N_{\text{crisis}}$) | Unweighted cross-entropy / Gini loss ($w_1 = w_0 = 1$) | Recall collapses to 0.0% (0 detections) across both LR and RF models at $\tau = 0.50$. | In severely imbalanced panels (2.7% prevalence), cost-sensitive reweighting is strictly mandatory to prevent complete majority-class collapse. |
| **3. Threshold Sensitivity** | Standard cutoff ($\tau = 0.50$) | Conservative ($\tau = 0.75$) and Aggressive ($\tau = 0.25$) cutoffs | At $\tau = 0.25$, LR catches 100% of crises (19/19) but generates 664 false alarms (FPR 99.8%). At $\tau = 0.75$, Recall is 0%. | Demonstrates the extreme policy trade-off: false alarms cannot be reduced without missing almost all crises. Thresholds must reflect policy cost functions. |
| **4. Expanding Fold Stability** | Pooled 2000–2008 aggregation | Fold-by-fold sequential expanding metrics | In 4 calm single-class folds (2000, 2003, 2004, 2005), AUC is Not Estimable. 15 of 19 crises cluster in 2007. | Early warning performance is episodic and driven by systemic global waves rather than steady-state annual incidence. |

*Table 4: Summary of Phase 5C empirical robustness and sensitivity evaluations.*

---

## 13. Temporal Generalization Diagnostic Analysis

While the pooled out-of-sample metrics in Table 3 indicate that the 6-variable Extended Logistic Regression outperforms the baseline, evaluating aggregate metrics across a pooled panel can obscure severe temporal concentration.

### The 2007 Crisis Clustering Phenomenon
An inspection of the 19 evaluation crisis events reveals an extreme temporal clustering:
- **15 of the 19 evaluation events (78.9%) occur in a single test fold: Test Year 2007** (predicting systemic banking crisis onsets in 2008).
- **Only 4 crisis events (21.1%) are distributed across the remaining 8 evaluation folds** ($T = 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2008$).

To determine whether the extended specification represents a broadly generalizable early warning signal or an empirical artifact of the 2007 pre-GFC episode, Phase 5E executed a dedicated temporal-generalization diagnostic. Crucially, this was conducted without retraining models, modifying features, or tuning thresholds. Table 5 reports the performance of Extended Logistic Regression across temporal slices.

| Temporal Subset | Evaluation Scope | Folds Count | Observations ($N$) | Crises ($N_{\text{pos}}$) | PR-AUC | ROC-AUC | Recall ($\tau=0.50$) | Precision | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Out-of-Sample** | All Test Folds (2000–2008) | 9 | 684 | 19 | **0.0344** | **0.5734** | **47.37%** ($9/19$) | 4.59% | 0.0837 |
| **2007 GFC Test Fold** | Test Year 2007 ($t \to 2008$) | 1 | 76 | 15 | **0.2647** | **0.6470** | **60.00%** ($9/15$) | **31.03%** | **0.4091** |
| **Excluding 2007 Fold** | Folds 2000–2006, 2008 | 8 | 608 | 4 | **0.0051** | **0.2301** | **0.00%** ($0/4$) | **0.00%** | **0.0000** |
| **Pre-GFC Expansion** | Folds 2000–2006 | 7 | 532 | 3 | **0.0051** | **0.2936** | **0.00%** ($0/3$) | **0.00%** | **0.0000** |
| **Post-GFC Fold** | Test Year 2008 ($t \to 2009$) | 1 | 76 | 1 | 0.0139 | 0.0533 | 0.00% ($0/1$) | 0.00% | 0.0000 |

*Table 5: Temporal generalization breakdown of Extended Logistic Regression across out-of-sample slices.*

### The Breakdown Outside the 2007 Fold
The diagnostic results in Table 5 reveal a dramatic divergence in predictive capability:
1. **High Detection in the 2007 Epicenter Fold:** In the 2007 test fold ($N=76$, 15 crises), the Extended Logistic Regression achieves strong performance: PR-AUC reaches $0.2647$, ROC-AUC reaches $0.6470$, Precision rises to $31.03\%$, and Recall reaches **60.00%** (detecting 9 of 15 crisis economies: Denmark, Spain, France, Greece, Hungary, Ireland, Iceland, Italy, and Portugal).
2. **Complete Collapse in Non-2007 Folds:** When the 2007 test fold is excluded, leaving 608 observations and 4 crisis events, the model experiences a complete breakdown in predictive discrimination:
   - **Recall collapses to 0.00% ($0/4$).** The model misses 100% of non-2007 crisis episodes.
   - **PR-AUC drops to $0.0051$,** falling below the unconditional non-2007 sample base rate ($4 / 608 = 0.0066$).
   - **ROC-AUC inverts severely to $0.2301$.** An ROC-AUC substantially below 0.50 demonstrates an observed ranking inversion: the model systematically assigned *lower* crisis probabilities to the countries that actually experienced crises than to the tranquil panel observations.
3. **Statistical Inversion Mechanism:** In the tranquil non-crisis panel ($N=665$), the median predicted probability generated by Extended LR is **0.4733** (mean = 0.4874). Within the four observed non-2007 crisis events, the median predicted probability is **0.4411** (mean = 0.4109; Uruguay 0.4264, Dominican Republic 0.4635, United Kingdom 0.4559, Nigeria 0.2978). Because the model assigned systematically lower probabilities to non-2007 crisis events than to calm panel observations, the observed ranking inverted.

### Approved Scientific Formulation
These empirical findings prohibit any claim that the 6-variable model represents a universal crisis forecasting engine. My interpretation is straightforward:

> *"The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal."*

---

## 14. Crisis Mechanism Analysis: A Three-Layer Framework

To explain why the Extended Logistic Regression successfully detected 9 crisis economies in 2007 but missed 6 economies in 2007 and all 4 economies outside 2007, Phase 6 instituted a formal **three-layer mechanism architecture** (Figure 2).

```
+----------------------------------------------------------------------------------------------------+
| LAYER A: AUTHORITATIVE GROUND TRUTH (IMF WP/26/94)                                                 |
| - 19 Systemic Banking Crises confirmed as non-borderline events.                                   |
| - Official Sub-Mechanism Taxonomy: "Unknown / insufficient evidence" (IMF does not classify types).|
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
| LAYER B: DATA-DRIVEN DESCRIPTIVE TYPOLOGY (Observed Predictors at Year t)                          |
| - Descriptive Profile A (Detected 2007, N=9): High Credit (142.2% GDP), High Deficit (-7.29% GDP). |
| - Descriptive Profile B (Missed 2007, N=6): High Credit (103.4% GDP), Large Surplus (+6.19% GDP).   |
| - Descriptive Profile C (Missed Non-2007, N=4): Lower Credit (53.9% GDP), Higher Inflation (4.79%).|
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
| LAYER C: NON-CAUSAL ECONOMIC HYPOTHESES (Fitted Model Logit Weight Alignment)                      |
| - Large positive credit weight (beta = +0.5142) elevated Profile A risk scores.                    |
| - Negative current account weight (beta = -0.0463) damped Profile B risk scores below 0.50 cutoff. |
| - Missed non-2007 events lacked credit-overhang signature, keeping probabilities suppressed.      |
+----------------------------------------------------------------------------------------------------+
```
*Figure 2: The Phase 6 Three-Layer Mechanism Architecture.*

### Layer A: Authoritative Ground Truth
Official IMF source documentation (Laeven & Valencia, 2026) verifies that all 19 evaluation events were non-borderline systemic banking crises. However, the IMF chronology does not provide a validated sub-mechanism classification taxonomy (e.g., distinguishing real-estate credit booms from interbank contagion or sovereign debt spillovers). Therefore, to preserve strict research integrity, all 19 crisis events are authoritatively recorded as **"Unknown / insufficient evidence"** at the ground-truth layer.

### Layer B: Data-Driven Descriptive Typology
At the descriptive layer, an inspection of observable predictor profiles at year $t$ partitions the 19 evaluation crises into three distinct macro-financial configurations:
1. **Descriptive Profile A: High Domestic Credit Overhang & External Deficit ($N=9$ Detected in 2007):**
   - Economies: Denmark, Spain, France, Greece, Hungary, Ireland, Iceland, Italy, Portugal.
   - Descriptive Characteristics: Median private credit to GDP = **142.2%**; median current account balance = **-7.29% of GDP** (all 9 ran deficits, reaching -14.19% in Greece and -13.63% in Iceland).
   - Probability Behavior: All predicted probabilities $\ge 0.50$ (median $\hat{p} = 0.5222$, range $[0.5018, 0.5701]$). All 9 detected.
2. **Descriptive Profile B: High Current Account Surplus & Wholesale Exposure ($N=6$ Missed in 2007):**
   - Economies: Austria, Belgium, Switzerland, Germany, Netherlands, Sweden.
   - Descriptive Characteristics: Median private credit to GDP = **103.4%**; median current account balance = **+6.19% of GDP** (every single economy ran a substantial surplus: Switzerland +8.63%, Sweden +8.15%, Germany +6.52%, Netherlands +5.86%, Austria +3.81%, Belgium +1.50%).
   - Probability Behavior: Predicted probabilities cluster tightly in the narrow band $[0.4445, 0.4795]$ (median $\hat{p} = 0.4677$), falling just short of the 0.50 classification cutoff. All 6 missed.
3. **Descriptive Profile C: Non-2007 Heterogeneous Shocks ($N=4$ Missed Outside 2007):**
   - **Uruguay (evaluated $t=2001 \to \text{onset } 2002$):** Credit/GDP = 53.9%, Current Account = -2.38%, Inflation = 4.36%, GDP Growth = -3.84%. Predicted $\hat{p} = 0.4264$.
   - **Dominican Republic (evaluated $t=2002 \to \text{onset } 2003$):** Credit/GDP = missing (imputed 53.9%), Current Account = -2.94%, Reserve Growth = -57.02%. Predicted $\hat{p} = 0.4635$.
   - **United Kingdom (evaluated $t=2006 \to \text{onset } 2007$):** Credit/GDP = 153.5%, Current Account = -3.11%, Inflation = 2.46%. Predicted $\hat{p} = 0.4559$.
   - **Nigeria (evaluated $t=2008 \to \text{onset } 2009$):** Credit/GDP = 18.6%, Current Account = +8.59%, Inflation = 11.58%, GDP Growth = +6.76%. Predicted $\hat{p} = 0.2978$.
   - Probability Behavior: Predicted probabilities range from $0.2978$ to $0.4635$ (median $\hat{p} = 0.4411$). All 4 missed.

### Layer C: Non-Causal Economic Hypotheses
The interaction between the fitted model parameters and the descriptive profiles explains the empirical results:
- In the Extended Logistic Regression, domestic private credit carries a large positive weight ($\beta = +0.5142$), while the current account balance carries a negative weight ($\beta = -0.0463$, meaning surpluses reduce crisis log-odds).
- For **Profile A economies**, the combination of elevated credit leverage and deep current account deficits pushed predicted probabilities above 0.50.
- For **Profile B economies**, despite high banking leverage, large current account surpluses exerted a damping effect on predicted log-odds, keeping probabilities below 0.50. Historical narratives indicate that banks in Germany, Switzerland, and Belgium suffered systemic distress due to off-balance-sheet exposures to US subprime securities and cross-border wholesale interbank contagion—exposures completely invisible to domestic aggregate current account balances.
- For **Profile C economies**, the crises involved idiosyncratic mechanisms (e.g., severe depositor contagion from Argentina in Uruguay; massive commercial bank fraud in the Dominican Republic; oil price shocks and margin lending in Nigeria). Because these episodes lacked the trans-Atlantic pre-GFC credit-overhang profile, the model failed to issue early warning signals.

---

## 15. Primary Model Interpretation and Parameter Estimates

Table 6 reports the parameter estimates, standard errors, odds ratios, and non-causal associative interpretations for the Primary Research Specification (Extended Logistic Regression) estimated on the final expanding training fold ($T=2008$, trained on 1990–2007, $N_{\text{train}} = 1,368$).

| Feature Name | Feature Label | Logit Coeff ($\beta$) | Odds Ratio ($\text{OR}$) | Direction | RF Importance | Cautious Associative Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `private_credit_pct_gdp` | Domestic Private Credit (% of GDP) | **+0.5142** | **1.6723** | Positive | 22.46% | Holding other included predictors constant, the fitted model associates a 1 SD increase in domestic private credit-to-GDP with a 67.2% increase in the estimated odds of a crisis onset in $t+1$ ($\text{OR} = 1.6723$). |
| `reserves_usd` | Gross Reserves (USD Level) | **-0.1708** | **0.8430** | Negative | 13.34% | Holding other included predictors constant, the fitted model associates larger reserve stocks with a 15.7% decrease in the estimated odds of crisis onset ($\text{OR} = 0.8430$). |
| `gdp_growth` | Real GDP Growth (Annual %) | **-0.1277** | **0.8801** | Negative | 9.01% | Holding other included predictors constant, the fitted model associates higher real GDP growth with a 12.0% decrease in the estimated odds of crisis onset ($\text{OR} = 0.8801$). |
| `inflation` | CPI Inflation Rate (Annual %) | **+0.0963** | **1.1011** | Positive | 18.33% | Holding other included predictors constant, the fitted model associates higher consumer price inflation with a 10.1% increase in the estimated odds of crisis onset ($\text{OR} = 1.1011$). |
| `current_account_pct_gdp` | Current Account Balance (% of GDP) | **-0.0463** | **0.9547** | Negative | 11.06% | Holding other included predictors constant, the fitted model associates a 1 SD increase in current account balance (surplus) with a 4.5% decrease in estimated crisis odds ($\text{OR} = 0.9547$; deficits elevate odds). |
| `reserves_usd_yoy_pct_change`| Reserves YoY Growth Rate (%) | **-0.0449** | **0.9561** | Negative | 25.79% | Holding other included predictors constant, the fitted model associates positive annual reserve growth with a 4.4% decrease in estimated crisis odds ($\text{OR} = 0.9561$). |

*Table 6: Parameter estimates and non-causal interpretations for the Primary Research Specification (Extended Logistic Regression, Fold $T=2008$).*

### Methodological Rules for Parameter Interpretation
To prevent misinterpretation, four strict econometric principles apply to Table 6:
1. **Standardized Predictors:** Because features were normalized to unit variance prior to estimation, coefficients represent log-odds shifts per one-standard-deviation change in the underlying indicator.
2. **Associative, Not Causal:** Coefficients and odds ratios reflect observational correlations in historical panel data. They do NOT establish that an exogenous increase in private credit *causes* a financial crisis.
3. **Training Partition Estimates:** These parameters reflect the model fitted on historical training data through 2007. They are not out-of-sample causal estimates.
4. **Monotonicity Check:** The signs of all six coefficients align with classical macroeconomic theory: credit expansions and inflation elevate crisis probabilities; reserve stocks, reserve growth, GDP growth, and current account surpluses reduce crisis probabilities.

---

## 16. Discussion

The empirical findings from this investigation yield several critical insights for economic policymakers, macroprudential authorities, and financial data scientists:

### 1. The Policy Value and Specificity of Balance-Sheet Stock Indicators
My findings support what macroeconomists emphasize that short-term macroeconomic flows cannot adequately reflect accumulating financial fragility. As demonstrated by the comparative results in Table 3, incorporating balance-sheet leverage and external financing imbalances substantially strengthens the identification of vulnerabilities in economies undergoing classic pre-crisis credit expansions, such as those observed across the periphery of the euro area prior to 2008. By capturing the accumulation of stock imbalances—specifically domestic credit depth paired with external borrowing—the extended specification successfully flags economies vulnerable to sudden liquidity freezes. However, this indicator set is inherently specific to debt-accumulation regimes; it does not capture banking distress originating from sovereign debt spillovers, sudden terms-of-trade collapses, or bank-level governance failures.

### 2. The Fallacy of Pooled Metrics and the Necessity of Temporal Slicing
The central methodological takeaway of this investigation is the hazard of relying exclusively on pooled out-of-sample metrics in rare-event panel contexts. If evaluated strictly on pooled metrics, the extended specification appears to offer a universally superior early warning signal. Yet granular temporal slicing reveals that pooled performance gains can be entirely concentrated within a single historical wave. In highly clustered event environments, pooled metrics conflate regime-specific identification with cross-regime forecasting ability. For macroprudential authorities, this demonstrates that validation protocols must explicitly test temporal stability across distinct historical episodes to prevent false confidence in model generalizability.

### 3. Machine Learning Ensembles and the Probability Compression Dilemma
A common assumption in applied predictive modeling is that non-linear ensemble architectures will naturally outperform generalized linear models by discovering complex interaction boundaries. In my empirical setting, however, Random Forests failed to provide useful early warning signals. Under severe class imbalance, tree ensembles minimize aggregate squared error by shrinking predicted probabilities toward the empirical base rate. Consequently, predicted risks rarely reach operational policy thresholds. In contrast, linear logistic regression, combined with cost-sensitive loss reweighting and monotonic log-odds scaling, permits risk scores to elevate during credit booms. For policy early warning under extreme skewness, parametric transparency and proper loss weighting prove vastly more critical than algorithmic complexity.

### 4. The Inherent False-Alarm Trade-Off in Policy Surveillance
Early warning models inevitably confront a trade-off between crisis detection sensitivity and false-alarm frequency. At standard classification thresholds, achieving meaningful recall requires tolerating a low precision environment where false alarms substantially outnumber true detections. In operational central banking, false alarms entail non-trivial welfare costs, such as unnecessary macroprudential tightening or market anxiety. My sensitivity analyses confirm that false alarms cannot be suppressed without missing the vast majority of systemic events. Quantitative early warning systems must therefore be deployed as broad screening mechanisms that motivate deeper supervisory scrutiny, rather than automated triggers for discretionary policy intervention.

---

## 17. Methodological Limitations

To maintain strict academic integrity, this research acknowledges eleven structural limitations:

1. **Rare-Event Small Sample Size ($N=19$ Total, $N=4$ Non-2007):** The out-of-sample evaluation window contains only 19 positive crisis events across nine years, with only 4 events occurring outside 2007. This small sample size imposes severe statistical limits on cross-regime generalization claims.
2. **Historical Evaluation Window and Holdout Availability:** The expanding-window evaluation period terminates at 2008. The post-2008 panel contains no positive evaluation holdouts in my sample, meaning this framework has not been tested against modern, post-Basel III banking environments.
3. **Revised WDI Data vs. Real-Time Vintages:** The World Bank WDI indicators utilized in this study represent revised historical time series rather than real-time data vintages available at each historical decision point. In real-time surveillance, national accounts data are subject to substantial publication lags (often 6 to 18 months) and subsequent statistical revisions.
4. **Missing Cross-Border Wholesale and Interbank Exposures:** As demonstrated by the missed 2007 crises in Germany, Switzerland, Belgium, and the Netherlands, national macroeconomic aggregates cannot capture off-balance-sheet vehicles, cross-border wholesale contagion, or foreign asset exposures. Incorporating Bank for International Settlements (BIS) consolidated banking statistics and bilateral interbank exposure networks is an essential next step.
5. **Annual Frequency and Intra-Year Liquidity Dynamics:** Annual temporal granularity cannot capture rapid intra-year liquidity freezes, abrupt depositor runs, or emergency policy interventions that unfold over days or weeks (e.g., the run on Northern Rock in September 2007).
6. **Severe Class Imbalance and False-Alarm Burden:** With an unconditional crisis prevalence of only 2.78%, achieving meaningful detection sensitivity requires tolerating a substantial volume of false alarms (187 false alarms generated by the primary model, yielding a precision of 4.59%). In practical policy settings, false alarms impose non-trivial economic and political costs.
7. **Country Heterogeneity Across Advanced and Emerging Economies:** Pooling 28 advanced economies and 48 emerging economies within a single parameter vector assumes structural parameter homogeneity across economies with vastly different institutional depths, exchange rate regimes, and financial development tiers.
8. **Restricted Six-Variable Predictor Set:** The empirical specification is restricted to six multilateral macro-financial variables. Relevant leading indicators—such as property price indices, bank capital adequacy ratios, sovereign credit default swap (CDS) spreads, and financial condition indices—were excluded to maintain historical panel completeness.
9. **No Causal Identification:** All reported relationships represent observational associations within historical panel data. The framework does not claim that private credit expansions or external deficits exogenously caused banking crises, nor can parameters be interpreted as structural policy multipliers.
10. **Probability Calibration in Extreme Rare-Event Tails:** While Brier scores measure probabilistic accuracy, predicted probabilities from cost-sensitively weighted models should not be interpreted as objective, uncalibrated real-world crisis likelihoods without post-hoc Platt scaling or isotonic calibration.
11. **Non-Operational Research Baseline:** The framework serves as an empirical research and diagnostic baseline to evaluate information content across indicators, not as an operational, deployment-ready early warning system for live central bank surveillance.

---

## 18. Computational Reproducibility and Software Architecture

This project was developed under a strict commitment to computational reproducibility and scientific transparency. Every table, figure, metric, and finding reported in this paper can be deterministically reproduced from raw source data using the open-source repository codebase.

### Computational Environment and Repository Architecture
- **Language & Runtime:** Python 3.12.14 on macOS Darwin / POSIX-compliant environments.
- **Core Dependencies:** Python scientific ecosystem using `scikit-learn` (Pedregosa et al., 2011), `pandas`, `numpy`, `pytest`, `ruff`, `pyyaml`, `pillow`.
- **Packaging:** Standardized PEP 621 compliant configuration via `pyproject.toml`.
- **Repository Structure:**
  - `config/`: Declarative YAML configurations (`variables.yaml`, `countries.yaml`, `expanded_research.yaml`).
  - `src/crisis_ews/`: Modular source code spanning collectors, panel builders, preprocessing, models, and evaluation engines.
  - `tests/`: Automated unit test suite containing 80 verified test cases.
  - `results/tables/`: Versioned CSV output tables (whitelisted in `.gitignore`).
  - `results/figures/`: Publication-quality SVG and companion PNG figures.
  - `docs/`: Comprehensive methodology documents and the detailed chronological research log (`docs/research_log.md`).

### Reproducibility Commands
To execute the complete pipeline, verify code quality, and reproduce all Phase 7 synthesis artifacts:

```bash
# 1. Clone the repository
git clone https://github.com/ege211/global-financial-crisis-ews.git
cd global-financial-crisis-ews

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Execute the full final synthesis pipeline
python -m crisis_ews.cli run-final-synthesis

# 4. Execute the automated test suite (80 passed tests)
pytest -v

# 5. Execute code formatting and style verification
ruff check .
```

---

## 19. Conclusion

This research investigated whether macro-financial indicators can provide reliable early warning signals for systemic banking crises within an objective machine-learning framework. By enforcing an expanding-window out-of-sample temporal validation design, leak-free train-only preprocessing, and multi-criteria evaluation on an objective 76-country panel, the study establishes four central conclusions:

1. Incorporating domestic private credit depth and external current account balances into a macroeconomic flow baseline improves pooled out-of-sample crisis detection, increasing PR-AUC from 0.0278 to 0.0344 and doubling crisis recall from 21.05% to 47.37%.
2. Parametric linear Logistic Regression outperforms non-linear Random Forest ensembles in rare-event detection, as tree models struggle when crises are very rare, tending to guess low probabilities everywhere to minimize simple error.
3. The empirical gains from balance-sheet indicators are episodic: "The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal."
4. Outside the 2007 fold, the model detected zero crises across the four observed historical episodes, accompanied by an observed ranking inversion.

Consequently, the empirical evidence provides **limited support for a broadly generalizable early warning signal**. The extended specification functions as an effective diagnostic tool for domestic credit-boom and external-deficit vulnerabilities, but does not constitute a universal crisis forecasting engine. These findings highlight the fundamental importance of temporal validation and crisis heterogeneity, underscoring that early warning frameworks must be deployed as regime-qualified diagnostic aids rather than automated forecasting algorithms.

---

## References

1. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
2. Gourinchas, P. O., & Obstfeld, M. (2012). Stories of the twentieth century for the twenty-first. *American Economic Journal: Macroeconomics*, 4(1), 226–265.
3. Kaminsky, G. L., Lizondo, S., & Reinhart, C. M. (1998). Leading indicators of currency crises. *IMF Staff Papers*, 45(1), 1–48.
4. Kaminsky, G. L., & Reinhart, C. M. (1999). The twin crises: The causes of banking and balance-of-payments problems. *American Economic Review*, 89(3), 473–500.
5. Laeven, L., & Valencia, F. (2013). Systemic banking crises database. *IMF Economic Review*, 61(2), 225–270.
6. Laeven, L., & Valencia, F. (2018). Systemic banking crises revisited. *IMF Working Papers*, WP/18/206, 1–44.
7. Laeven, L., & Valencia, F. (2020). Systemic banking crises database II. *IMF Economic Review*, 68(2), 307–361.
8. Laeven, L., & Valencia, F. (2026). Systemic banking crises database: A 2026 update. *IMF Working Papers*, WP/26/94, 1–52.
9. Obstfeld, M. (2012). Does the current account still matter? *American Economic Review*, 102(3), 1–23.
10. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
11. Reinhart, C. M., & Rogoff, K. S. (2009). *This Time Is Different: Eight Centuries of Financial Folly*. Princeton, NJ: Princeton University Press.
12. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
13. Schularick, M., & Taylor, A. M. (2012). Credit booms gone bust: Monetary policy, leverage cycles, and financial crises, 1870–2008. *American Economic Review*, 102(2), 1029–1061.
14. World Bank. (2026). *World Development Indicators Database*. Washington, DC: World Bank Group.

---

## Appendix A — Model Comparison Table

*Full comparative metrics table across all four evaluated architectures based on [final_model_comparison.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_model_comparison.csv).*

| Metric Dimension | Baseline Logistic Regression | Extended Logistic Regression | Baseline Random Forest | Extended Random Forest |
| :--- | :---: | :---: | :---: | :---: |
| **Model Class** | Generalized Linear Model (Logit) | Generalized Linear Model (Logit) | Non-parametric Ensemble (100 Trees) | Non-parametric Ensemble (100 Trees) |
| **Predictor Count ($K$)** | 4 (Macro flows) | 6 (Flows + Balance sheet) | 4 (Macro flows) | 6 (Flows + Balance sheet) |
| **Out-of-Sample Observations ($N$)** | 684 | 684 | 684 | 684 |
| **Out-of-Sample Crisis Onsets ($N_{\text{pos}}$)** | 19 | 19 | 19 | 19 |
| **PR-AUC (Pooled 2000–2008)** | 0.0278 | **0.0344** | 0.0256 | 0.0274 |
| **ROC-AUC (Pooled 2000–2008)** | 0.5106 | **0.5734** | 0.4712 | 0.4715 |
| **Brier Score (Pooled 2000–2008)** | 0.2512 | 0.2482 | 0.1524 | **0.1347** |
| **Recall (Pooled, $\tau=0.50$)** | 21.05% ($4/19$) | **47.37%** ($9/19$) | 21.05% ($4/19$) | 10.53% ($2/19$) |
| **Precision (Pooled, $\tau=0.50$)** | 1.89% ($4/212$) | **4.59%** ($9/196$) | 2.42% ($4/165$) | 1.61% ($2/124$) |
| **F1-Score (Pooled, $\tau=0.50$)** | 0.0346 | **0.0837** | 0.0435 | 0.0280 |
| **True Positives (Pooled)** | 4 | **9** | 4 | 2 |
| **False Positives (Pooled)** | 208 | 187 | 161 | **122** |
| **False Negatives (Pooled)** | 15 | **10** | 15 | 17 |
| **True Negatives (Pooled)** | 457 | 478 | 504 | **543** |
| **2007 GFC Fold Recall ($N=15$)** | 20.00% ($3/15$) | **60.00%** ($9/15$) | 20.00% ($3/15$) | 13.33% ($2/15$) |
| **Non-2007 Folds Recall ($N=4$)** | **25.00%** ($1/4$) | 0.00% ($0/4$) | **25.00%** ($1/4$) | 0.00% ($0/4$) |
| **2007 GFC Fold ROC-AUC** | **0.6590** | 0.6470 | 0.3454 | 0.3552 |
| **Non-2007 Folds ROC-AUC** | 0.3845 | 0.2301 | **0.5555** | 0.5401 |
| **Interpretability Level** | High (Linear log-odds) | High (Linear log-odds) | Moderate (Gini importance) | Moderate (Gini importance) |
| **Selection Role** | Baseline Benchmark | **Primary Research Specification** | Nonlinear Benchmark | Exploratory ML Specification |
| **Selection Status** | Benchmark | **Retained (Regime-Qualified)** | Rejected | Rejected |

---

## Appendix B — Temporal Validation Table

*Sequential fold-by-fold expanding-window metrics for the Primary Research Specification (Extended Logistic Regression) across test folds $T=2000, \dots, 2008$ based on [expanded_extended_logistic_regression_fold_metrics.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/expanded_extended_logistic_regression_fold_metrics.csv).*

| Test Fold ($T$) | Training Window | Train $N$ | Test $N$ | Test Crises | Fold PR-AUC | Fold ROC-AUC | Fold Recall | Fold Precision | Fold Brier | TP | FP | FN | TN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2000** | 1990–1999 | 760 | 76 | 0 | Not Estimable | Not Estimable | 0.0% | 0.0% | 0.2396 | 0 | 18 | 0 | 58 |
| **2001** | 1990–2000 | 836 | 76 | 1 | 0.0161 | 0.2933 | 0.0% | 0.0% | 0.2390 | 0 | 14 | 1 | 61 |
| **2002** | 1990–2001 | 912 | 76 | 1 | 0.0172 | 0.4400 | 0.0% | 0.0% | 0.2443 | 0 | 18 | 1 | 57 |
| **2003** | 1990–2002 | 988 | 76 | 0 | Not Estimable | Not Estimable | 0.0% | 0.0% | 0.2464 | 0 | 25 | 0 | 51 |
| **2004** | 1990–2003 | 1,064 | 76 | 0 | Not Estimable | Not Estimable | 0.0% | 0.0% | 0.2367 | 0 | 17 | 0 | 59 |
| **2005** | 1990–2004 | 1,140 | 76 | 0 | Not Estimable | Not Estimable | 0.0% | 0.0% | 0.2359 | 0 | 16 | 0 | 60 |
| **2006** | 1990–2005 | 1,216 | 76 | 1 | 0.0169 | 0.2400 | 0.0% | 0.0% | 0.2520 | 0 | 24 | 1 | 51 |
| **2007** | 1990–2006 | 1,292 | 76 | 15 | **0.2647** | **0.6470** | **60.0%** | **31.03%** | 0.2371 | **9** | 20 | 6 | 41 |
| **2008** | 1990–2007 | 1,368 | 76 | 1 | 0.0139 | 0.0533 | 0.0% | 0.0% | 0.2979 | 0 | 35 | 1 | 40 |
| **Total / Pooled** | — | — | **684** | **19** | **0.0344** | **0.5734** | **47.37%** | **4.59%** | **0.2482** | **9** | **187** | **10** | **478** |

---

## Appendix C — Robustness Results Summary

*Comprehensive synthesis of evaluated robustness dimensions based on [final_robustness_summary.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_robustness_summary.csv).*

| Robustness Dimension | Evaluated Focus | Baseline Specification | Perturbation Condition | Empirical Outcome | Econometric Implication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Winsorization Sensitivity** | Sensitivity to extreme tail inflation outliers | Raw CPI inflation (untrimmed) | CPI inflation winsorized at [1st, 99th] percentiles across training folds | Logistic PR-AUC drops from 0.0278 to 0.0219 (-21.2%), ROC-AUC drops from 0.5106 to 0.3893, Recall collapses from 21.05% to 5.26% (1 detection); Random Forest metrics remain identical (PR-AUC 0.0256, Recall 21.05%). | Linear logit models rely heavily on extreme inflation tails for rare-event separation; truncating outliers destroys predictive signal. Tree models are inherently invariant to monotonic outlier shifts. |
| **Class Weight Sensitivity** | Impact of cost-sensitive loss weighting | Balanced class weighting ($w_1 \propto 1/N_{\text{crisis}}$) | Unweighted cross-entropy / Gini loss ($w_1 = w_0 = 1$) | Both LR and RF collapse to 0.0% Recall (0 true positives, 0 false alarms) at standard 0.50 cutoff, predicting calm state for 100% of observations. | In severely imbalanced panels (2.7% positive rate), cost-sensitive loss weighting is strictly mandatory to prevent default majority-class convergence. |
| **Threshold Sensitivity** | Trade-off between detection and false alarms | Standard cutoff ($\tau = 0.50$) | Conservative ($\tau = 0.75$) and Aggressive ($\tau = 0.25$) cutoffs | At $\tau = 0.25$, LR achieves 100% Recall (19/19) but issues 664 false alarms (FPR 99.8%); at $\tau = 0.75$, LR achieves 0% Recall with 15 false alarms. | Demonstrates severe policy trade-off; early warning systems cannot eliminate false alarms without missing almost all crises. Threshold selection must be loss-function-driven. |
| **Expanding Fold Stability** | Temporal stability across 9 sequential test years | Pooled 2000–2008 aggregation | Fold-by-fold expanding metrics evaluation | In 4 calm single-class folds (2000, 2003, 2004, 2005), AUC metrics are Not Estimable; 15 of 19 positive events cluster in fold 2007 (forecasting 2008). | Early warning performance is episodic and driven by multi-country systemic waves. Out-of-sample metrics are not steady-state. |
| **Predictor Expansion** | Value of adding balance-sheet indicators | 4-variable macroeconomic flow baseline | 6-variable extended set (+credit/GDP, +current account/GDP) | Pooled PR-AUC increases from 0.0278 to 0.0344 (+23.7%), ROC-AUC increases from 0.5106 to 0.5734, Recall surges from 21.05% to 47.37% (+125%), false positives decrease from 208 to 187. | Adding domestic private credit leverage and external deficits provides substantially stronger empirical discrimination than flow indicators alone. |
| **Regime Generalization** | Generalizability outside the 2007 pre-GFC episode | Pooled 2000–2008 evaluation window | Subsets excluding 2007 fold (N=4), pre-GFC 2000–2006 (N=3), and 2007 fold alone (N=15) | Excluding 2007, Extended LR Recall drops to 0.0% (0/4) and ROC-AUC inverts to 0.2301; in 2007, PR-AUC is 0.2647 and Recall is 60.0% (9/15). | The extended specification is regime-dependent. It detects private credit overhang paired with external deficits, but does not generalize to idiosyncratic crises. |

---

## Appendix D — Out-of-Sample Crisis Event Profiles

*Complete enumeration of all 19 out-of-sample crisis events evaluated in test folds $T=2000, \dots, 2008$ based on [phase6_crisis_event_profiles.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_crisis_event_profiles.csv).*

| ISO-3 | Country Name | Eval Fold ($T$) | Crisis Onset ($t+1$) | Extended $\hat{P}$ | Ext Class ($\tau=0.5$) | Extended Status | Baseline $\hat{P}$ | Baseline Status | Credit (% GDP) | Current Acc (% GDP) | Inflation (%) | GDP Growth (%) | Descriptive Typology Profile |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `URY` | Uruguay | 2001 | 2002 | 0.4264 | 0 | Missed | 0.4450 | Missed | 53.85% | -2.38% | 4.36% | -3.84% | Profile C (Non-2007 Idiosyncratic / EM) |
| `DOM` | Dominican Rep. | 2002 | 2003 | 0.4635 | 0 | Missed | 0.4655 | Missed | 53.85%* | -2.94% | 5.22% | +4.49% | Profile C (Non-2007 Idiosyncratic / EM) |
| `GBR` | United Kingdom | 2006 | 2007 | 0.4559 | 0 | Missed | 0.5046 | **Detected** | 153.47% | -3.11% | 2.46% | +2.19% | Profile C (Non-2007 Advanced Wholesale) |
| `AUT` | Austria | 2007 | 2008 | 0.4445 | 0 | Missed | 0.4576 | Missed | 93.50% | **+3.81%** | 2.17% | +3.78% | Profile B (2007 Surplus / Wholesale) |
| `BEL` | Belgium | 2007 | 2008 | 0.4659 | 0 | Missed | 0.4763 | Missed | 68.44% | **+1.50%** | 1.82% | +3.68% | Profile B (2007 Surplus / Wholesale) |
| `CHE` | Switzerland | 2007 | 2008 | 0.4730 | 0 | Missed | 0.4943 | Missed | 151.06% | **+8.63%** | 0.73% | +3.98% | Profile B (2007 Surplus / Wholesale) |
| `DEU` | Germany | 2007 | 2008 | 0.4795 | 0 | Missed | 0.5049 | **Detected** | 95.50% | **+6.52%** | 2.30% | +2.89% | Profile B (2007 Surplus / Wholesale) |
| `DNK` | Denmark | 2007 | 2008 | **0.5163** | **1** | **Detected** | 0.4990 | Missed | 184.04% | +1.51% | 1.69% | +0.99% | Profile A (2007 Credit Overhang) |
| `ESP` | Spain | 2007 | 2008 | **0.5619** | **1** | **Detected** | 0.5029 | **Detected** | 167.64% | **-9.37%** | 2.79% | +3.53% | Profile A (2007 Credit Overhang / Deficit) |
| `FRA` | France | 2007 | 2008 | **0.5078** | **1** | **Detected** | 0.5055 | **Detected** | 88.97% | -0.33% | 1.49% | +2.53% | Profile A (2007 Credit Overhang) |
| `GRC` | Greece | 2007 | 2008 | **0.5305** | **1** | **Detected** | 0.4687 | Missed | 85.80% | **-14.19%** | 2.90% | +3.51% | Profile A (2007 Credit Overhang / Deficit) |
| `HUN` | Hungary | 2007 | 2008 | **0.5222** | **1** | **Detected** | 0.4990 | Missed | 53.20% | **-7.29%** | 7.96% | +0.33% | Profile A (2007 Credit Overhang / Deficit) |
| `IRL` | Ireland | 2007 | 2008 | **0.5215** | **1** | **Detected** | 0.4821 | Missed | 158.10% | **-5.13%** | 4.89% | +5.31% | Profile A (2007 Credit Overhang / Deficit) |
| `ISL` | Iceland | 2007 | 2008 | **0.5701** | **1** | **Detected** | 0.4737 | Missed | 243.15% | **-13.63%** | 5.05% | +8.74% | Profile A (2007 Credit Overhang / Deficit) |
| `ITA` | Italy | 2007 | 2008 | **0.5018** | **1** | **Detected** | 0.4964 | Missed | 81.45% | -1.36% | 1.83% | +1.46% | Profile A (2007 Credit Overhang) |
| `NLD` | Netherlands | 2007 | 2008 | 0.4694 | 0 | Missed | 0.4885 | Missed | 112.72% | **+5.86%** | 1.61% | +3.89% | Profile B (2007 Surplus / Wholesale) |
| `PRT` | Portugal | 2007 | 2008 | **0.5400** | **1** | **Detected** | 0.4846 | Missed | 142.24% | **-9.66%** | 2.45% | +2.51% | Profile A (2007 Credit Overhang / Deficit) |
| `SWE` | Sweden | 2007 | 2008 | 0.4637 | 0 | Missed | 0.4931 | Missed | 111.25% | **+8.15%** | 2.21% | +3.22% | Profile B (2007 Surplus / Wholesale) |
| `NGA` | Nigeria | 2008 | 2009 | 0.2978 | 0 | Missed | 0.4865 | Missed | 18.63% | **+8.59%** | 11.58% | +6.76% | Profile C (Non-2007 Idiosyncratic / EM) |

*\*Note: Private credit for Dominican Republic in 2002 was missing in raw WDI and imputed using the fold training median ($53.85\%$). All sub-mechanisms are authoritatively classified as "Unknown / insufficient evidence" under Layer A.*

---

## Appendix E — Chronological Research Phase Audit Trail

The research presented in this paper was conducted across eight distinct, sequential research phases. The audit trail below documents the methodological locks, CLI commands, and verification checkpoints:

- **Phase 0–1 (Foundation & Ingestion Audit):** Established raw API collectors and verified that all historical data derived from World Bank WDI and IMF WP/26/94 sources. Confirmed zero initial look-ahead leakage.
- **Phase 2–3 (Architecture & Runbook Verification):** Constructed modular directory structure (`src/crisis_ews/`) and established immutable baseline configurations.
- **Phase 4A–4B (Universe Expansion & Data Audit):** Expanded candidate country universe to an objective 76-country balanced panel spanning 1990–2025. Verified coverage across 2,736 country-years and audited missingness patterns.
- **Phase 5A (Pre-Fit Audit & Validation Freeze):** Established expanding-window validation protocol ($T = 2000, \dots, 2008$). Verified that training sets condition strictly on $[1990, T-1]$, confirmed zero test-fold contamination, and locked all decision thresholds at $\tau = 0.50$.
- **Phase 5B (Baseline Model Estimation):** Fitted Baseline Logistic Regression and Baseline Random Forest across the 4 macroeconomic flow predictors. Established benchmark metrics ($N=684$, Positives=19, Baseline LR PR-AUC=0.0278, ROC-AUC=0.5106, Recall=21.05%).
- **Phase 5C (Robustness & Sensitivity Analysis):** Evaluated winsorization, unweighted loss training, threshold sensitivity ($\tau \in [0.10, 0.90]$), and sequential fold stability. Demonstrated necessity of cost-sensitive class weighting.
- **Phase 5D (Predictor Selection & Extended Estimation):** Formally selected and locked `private_credit_pct_gdp` and `current_account_pct_gdp` as the two additions to create the 6-variable specification. Estimated Extended LR (PR-AUC=0.0344, ROC-AUC=0.5734, Recall=47.37%) and Extended RF (Recall=10.53%).
- **Phase 5E (Temporal Generalization Diagnostic):** Discovered that 15 of 19 evaluation events clustered in the 2007 fold. Unmasked the complete breakdown outside 2007 (0.0% recall across the 4 non-2007 crisis episodes, ROC-AUC=0.2301 ranking inversion). Formulated the approved non-causal conclusion.
- **Phase 6 (Crisis Mechanism Analysis):** Instituted the three-layer mechanism framework. Documented Profile A (detected credit booms with deficits), Profile B (missed surplus economies), and Profile C (non-2007 episodes). Confirmed authoritative sub-mechanisms as "Unknown / insufficient evidence".
- **Phase 7 (Final Model Selection & Research Synthesis):** Formally retained the 6-variable Extended Logistic Regression as the Primary Research Specification with explicit regime qualifications. Generated consolidated synthesis tables (`final_model_comparison.csv`, `final_robustness_summary.csv`, `final_model_coefficients.csv`) and Figure 4.
- **Phase 8A (Academic Research Paper):** Authored `docs/RESEARCH_PAPER.md` as an admissions-quality, 27-section comprehensive academic research paper synthesizing all findings with zero causal overclaiming, exact numerical tracking, and complete computational reproducibility.
