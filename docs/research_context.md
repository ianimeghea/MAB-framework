# Research Context: Multi-Agent Bandit Learning in Dark Liquidity Pools

*Anchoring the MAB framework in a real financial problem and reviewing the state of the literature.*

---

## 1. The Real Finance Problem

### 1.1 Institutional order execution and the venue-selection problem

When a large institutional investor — a pension fund, hedge fund, or asset manager — needs to buy or sell a significant block of shares (say, $50 million of Apple stock), executing the order on a public exchange is dangerous. Posting a large buy order on NYSE or NASDAQ reveals the institution's intent to the whole market. High-frequency traders and other participants adjust their prices accordingly, and the institution ends up paying more than the pre-trade price. This cost is called **market impact** or **adverse selection**, and it is the central cost-minimisation problem in institutional trading.

Dark pools emerged as a partial solution. A **dark liquidity pool** is a private electronic trading venue in which orders are not visible before execution. The institution submits an order, and if the pool finds a counterparty willing to trade at or near the current midpoint, a fill is generated. Because the order is never displayed, there is no opportunity for other participants to react to it.

The problem, however, is that there are now dozens of dark pools operating simultaneously, each with different:
- **Fill rates**: the probability that an order submitted to this pool actually executes
- **Price improvement**: how much better than the midpoint the fill is
- **Adverse selection risk**: the probability that, conditional on being filled, the stock price moves against you afterwards

A **Smart Order Router (SOR)** is the algorithm that decides, for each order, which venue(s) to send it to. This is a high-dimensional, time-varying, partially-observable decision problem. The SOR must:
1. Learn the fill quality of each venue from historical execution data
2. Adapt to changing conditions (liquidity in pools changes intraday and over time)
3. Account for the fact that other institutions are routing to the same pools simultaneously

That third point is the one this framework addresses. **This is exactly a multi-agent multi-armed bandit problem.**

### 1.2 Mapping to the MAB framework

| Real-world element | MAB abstraction |
|---|---|
| Dark pool / venue | Arm |
| Execution quality (price improvement, fill rate) | Arm reward (mean, variance) |
| Submitting an order to a venue | Pulling an arm |
| Only observing your own fill outcome | Bandit feedback |
| Multiple institutions routing simultaneously | Multiple agents |
| Two institutions target the same pool | Collision |
| Pro-rata fill split (IEX, crossing networks) | `linear_share` |
| Pool detects crowded flow, cancels all fills | `zero_on_collision` |
| Fastest order wins the fill (latency priority) | `winner_takes_all` |
| Venue quality changing over time | Non-stationary arms |
| Routing decision at each timestep | One arm pull per step |

The mapping is not merely a convenient analogy. **Bandit feedback is structurally correct for dark pools**: because the pool hides its order book pre-trade, you cannot observe the counterfactual (what fill you would have received at any other venue). You only observe your own result. This is the defining property of a bandit problem — it is not a simplification.

### 1.3 Why the collision mechanism matters

The key informational feature of dark pools — absent from lit markets and from most MAB applications — is the **crowding externality**. When multiple institutions simultaneously route large orders to the same pool:

- In **lit markets** (NASDAQ, NYSE), simultaneous orders queue sequentially. No fill is cancelled; everyone gets executed in order.
- In **dark pools**, the outcome depends on the pool's mechanism:
  - A **pro-rata pool** (IEX model) splits fills proportionally by order size. Total liquidity is shared, not destroyed.
  - A **screening pool** that detects concentrated demand may cancel all fills, interpreting simultaneous flow as informed or predatory. This is the adverse-selection detection mechanism.
  - A **latency-priority pool** fills the fastest order and leaves the rest with nothing. This is the model for HFT-adjacent venues.

These correspond directly to `linear_share`, `zero_on_collision`, and `winner_takes_all`. This is the justification for introducing the dark pool framing: the collision mechanism is a domain-specific structural feature with a precise real-world interpretation and regulatory implications.

### 1.4 Regulatory context

Under **MiFID II** (EU, effective January 2018) and **Reg NMS** (US), brokers are legally required to achieve **best execution** — they must demonstrate that their routing decisions achieve the best available outcome for the client. This means SOR systems are no longer optional; they are legally mandated optimisation algorithms.

MiFID II additionally introduced **volume caps** on dark pools (4% per single venue, 8% across all dark pools for any given stock), directly incentivising venue diversification. The empirical result — that adaptive SOR systems fragment order flow across venues — has now become a regulatory requirement, not just an optimisation outcome. Our RQ2 finding (emergent diversification from pure learning) is therefore normatively relevant: it suggests that even without explicit regulatory programming, adaptive algorithms would arrive at fragmented routing through trial and error.

---

## 2. Answers to the Research Questions

### RQ1: How does the collision mechanism affect total market welfare and individual agent performance?

**Short answer**: The mechanism determines whether collisions *transfer* or *destroy* value. Under pro-rata allocation, welfare is preserved regardless of how much agents herd. Under toxic-pool detection, welfare is destroyed at every collision, and the welfare loss scales linearly with agent count and collision rate. Under latency priority, welfare is preserved in expectation but concentrated in a single agent per collision event.

**Detailed answer**:

*Market welfare*. Our experiments (RQ1-A, RQ1-B) confirm that `linear_share` and `winner_takes_all` maintain near-identical total welfare regardless of agent count, because the collision mechanism does not destroy value — it redistributes it. The total reward across all agents in a colliding step is unchanged. Under `zero_on_collision`, welfare degrades sharply as agent count increases: with 8 agents competing over 5 arms, the collision rate exceeds 90% and welfare drops to near zero until agents learn to diversify.

This maps directly to the real finance distinction: *pro-rata dark pools are (in aggregate) efficient regardless of crowding*; *screening pools impose a real welfare cost on the collective, even if each individual pool operator considers the mechanism desirable*.

*Learning dynamics* (RQ1-C). Temporal welfare plots show that `zero_on_collision` has the worst welfare in the early phase (before agents learn to avoid collisions) but converges to near-`linear_share` efficiency once diversification emerges. This means the long-run efficiency gap between policies is smaller than the short-run gap — but the short-run gap is precisely the phase when real institutions are routing orders in a new market condition.

*Arm heterogeneity* (RQ1-D). The policy effect is strongest when arm quality is heterogeneous (wide gap scenario). When all venues are equally good, agents have no reason to herd on any particular arm, and collision rates are naturally low under all policies. The real-world implication: in a fragmented market with one or two clearly superior dark pools, the adverse-selection / crowding problem is most acute.

*Strategy robustness* (RQ1-E). The collision policy matters more than the routing strategy for welfare: even a random router achieves comparable aggregate welfare to UCB under `linear_share` (because herding is costless), but UCB recovers significantly faster under `zero_on_collision` (because it updates its arm estimates from the zero-reward collision signal and adapts).

### RQ2: Do learning agents start diversifying to avoid collision, even without communication?

**Short answer**: Yes — but the mechanism and speed depend critically on both the collision policy and the strategy. UCB agents spontaneously specialise on different venues under `zero_on_collision`. Under `linear_share`, diversification pressure is weak and simpler agents herd persistently.

**Detailed answer**:

*Emergent specialisation* (RQ2-B, RQ2-E). Under `zero_on_collision`, UCB agents converge to anchoring on different arms in the late phase. One agent locks onto the best pool; the other, having been repeatedly punished for colliding on it, drifts to the second-best pool. This is not explicitly programmed — it emerges purely from the bandit feedback signal. The mechanism is: collision → zero reward → arm value estimate pulled toward zero → UCB score for that arm temporarily suppressed → the less-dominant agent explores alternatives. Over time, a stable separation forms.

This mirrors the empirical observation in dark pool microstructure: well-known institutional investors often have "preferred venues" that vary across firms, a form of implicit specialisation that reduces the aggregate crowding cost. Our result provides a theoretical mechanism for how this could arise through adaptive routing.

*Policy determines the pressure* (RQ2-A, RQ2-D). Under `linear_share`, the collision signal is weak: agents still receive positive rewards when colliding, so the arm estimate for the best arm remains high for all agents. They continue to herd because herding is not punished. Under `zero_on_collision`, the signal is sharp: zero reward is an unambiguous negative signal. Diversification emerges faster and more completely.

*Strategy determines the speed* (RQ2-D). UCB reaches near-zero late-phase collision rates in under 500 steps. Epsilon-greedy with low epsilon (0.05) diversifies more slowly because it mostly exploits — once it settles on the best arm, it rarely re-explores alternatives, making it harder for the learning signal to propagate. Epsilon-greedy with high epsilon (0.20) diversifies faster but does so partially randomly rather than purposefully.

*Crowding amplifies the effect* (RQ2-C). At low crowding ratios (n_agents / n_arms < 0.6), all policies behave similarly — there is enough arm capacity that agents naturally spread. Above ratio ~0.6, the welfare cliff for `zero_on_collision` becomes severe, and diversification pressure intensifies. Real dark pool markets have crowding ratios that vary significantly by stock liquidity and market conditions, suggesting the regime matters.

**Implication for SOR design**. An adaptive SOR based on UCB will naturally learn to fragment its order flow across venues, even without being explicitly designed to do so, *provided* the dark pool's collision mechanism penalises concentrated flow. This gives a theoretical grounding for the empirical fragmentation patterns observed in real markets, and suggests that pool mechanism design (not just SOR algorithm design) is a lever for promoting efficient venue allocation.

---

## 3. Literature Review

### 3.1 Foundational multi-armed bandit theory

The multi-armed bandit problem was introduced by **Robbins (1952)**, who framed sequential decision-making as a tradeoff between exploiting known good options and exploring uncertain ones. The paper established the foundational vocabulary of the field.

The asymptotic lower bound on regret — the minimum cost any learning algorithm must incur — was established by **Lai and Robbins (1985)**. They proved that the expected number of suboptimal pulls must grow at least logarithmically in the number of rounds, and derived the Kullback-Leibler divergence as the quantity governing this bound. This result gives the theoretical floor against which all algorithms are measured.

The practical UCB (Upper Confidence Bound) framework — the algorithm implemented in this thesis — was formalised by **Auer, Cesa-Bianchi, and Fischer (2002)** in their landmark finite-time analysis. They proved that UCB1 achieves logarithmic regret uniformly over time without requiring prior knowledge of reward distributions, a major theoretical advance over earlier asymptotic results. The UCB principle of optimism in the face of uncertainty provides a principled exploration bonus that exactly corresponds to the exploration mechanism our agents use.

For a comprehensive survey of the field, **Bubeck and Cesa-Bianchi (2012)** provide a 122-page treatment covering both stochastic and adversarial bandits, variants, and extensions. This is the standard reference for anyone entering the field.

Thompson Sampling — an important Bayesian alternative to UCB — was introduced by **Thompson (1933)** in a short paper on comparing unknown probabilities. Despite its early origin, it was largely overlooked until modern empirical results and theoretical analyses (Agrawal, 2012) confirmed its near-optimal performance.

---

**Key papers:**

- Robbins, H. (1952). Some aspects of the sequential design of experiments. *Bulletin of the American Mathematical Society*, 58(5), 527–535.
- Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics*, 6(1), 4–22. [[ScienceDirect](https://www.sciencedirect.com/science/article/pii/0196885885900028)]
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2–3), 235–256. [[Springer](https://link.springer.com/article/10.1023/A:1013689704352)]
- Bubeck, S., & Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. *Foundations and Trends in Machine Learning*, 5(1), 1–122. [[arXiv](https://arxiv.org/abs/1204.5721)]
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika*, 25(3–4), 285–294. [[Oxford](https://academic.oup.com/biomet/article-abstract/25/3-4/285/200862)]

---

### 3.2 Multi-player and multi-agent multi-armed bandits

The extension of MAB to multiple competing agents is a substantially harder problem and has received significant attention over the past decade, largely motivated by cognitive radio and wireless channel allocation — but the results transfer directly to any competitive resource-sharing problem.

**Besson and Kaufmann (2018)** revisited the multi-player MAB model (originally motivated by cognitive radio) and characterised the collision model formally: when multiple players pull the same arm, they all receive zero reward. They established improved lower bounds for decentralised algorithms and introduced practical algorithms (RandTopM, MCTopM) that outperform prior work. Their Selfish algorithm, which requires no inter-player communication and no collision feedback, is the closest theoretical analogue to the agents in this framework.

**Boursier and Perchet (2019)** introduced SIC-MMAB (Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits), a decentralised algorithm that achieves near-centralized performance by *deliberately engineering collisions* as a communication channel between players — encoding information in the collision pattern itself. This is a striking result that inverts the problem: rather than minimising collisions, the algorithm uses them as signals. This paper established a key insight relevant to our RQ2: collisions are information, not just noise.

**Bistritz and Leshem (2018)** studied the fully decentralised case where players cannot observe whether a collision occurred (no sensing). They present a distributed algorithm achieving O(log²T) regret — the first poly-logarithmic result in this setting. The paper is relevant because it establishes that diversification and near-optimal performance are achievable even with very limited feedback, which is the regime most relevant to dark pool trading (where you do not observe others' actions).

**Boursier and Perchet (2024)** published a comprehensive survey of the multi-player bandit literature in JMLR, covering cooperative settings (where players share a social welfare objective), competitive settings (where players are selfish), and the evolution of collision models from pure coordination problems to game-theoretic settings. This survey is the state-of-the-art reference for the subfield.

A very recent line of work studies the **competitive** multi-player bandit where agents are fully selfish and studies whether Nash equilibria exist and how they compare to the social optimum (price of anarchy). Results show that selfish bandits can have an infinite price of anarchy — meaning competitive behaviour can destroy arbitrarily large fractions of social welfare. This directly motivates RQ1: the collision mechanism is a design choice that can either limit or amplify this welfare loss.

---

**Key papers:**

- Besson, L., & Kaufmann, E. (2018). Multi-player bandits revisited. *Proceedings of Algorithmic Learning Theory (ALT)*, Vol. 83. [[PMLR](https://proceedings.mlr.press/v83/besson18a.html)] [[arXiv](https://arxiv.org/abs/1711.02317)]
- Boursier, E., & Perchet, V. (2019). SIC-MMAB: Synchronisation involves communication in multiplayer multi-armed bandits. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. [[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/c4127b9194fe8562c64dc0f5bf2c93bc-Abstract.html)]
- Bistritz, I., & Leshem, A. (2018). Distributed multi-player bandits — a game of thrones approach. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. [[NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/c2964caac096f26db222cb325aa267cb-Abstract.html)]
- Boursier, E., & Perchet, V. (2024). A survey on multi-player bandits. *Journal of Machine Learning Research*, 25. [[JMLR](https://jmlr.org/papers/v25/22-0643.html)]

---

### 3.3 MAB and reinforcement learning applied to dark pools and trade execution

This is the most directly relevant literature to this thesis, and it is thin — which defines the research gap.

**Ganchev, Kearns, Nevmyvaka, and Vaughan (2009 / 2010)** wrote what is arguably the single most important precursor to this work. Published at UAI 2009 and reprinted in *Communications of the ACM* (2010), their paper "Censored Exploration and the Dark Pool Problem" formalises the dark pool allocation problem as a bandit learning problem under *censored feedback* — you only observe whether your order was filled, not what the fill rate would have been at venues you did not submit to. They introduce and analyse an algorithm that converges in polynomial time to a near-optimal allocation. Crucially, however, they study a **single-agent problem**: one institution routing an order across multiple dark pools, with no competing institutions in the model. The multi-agent collision dimension — which is the central focus of this thesis — is entirely absent. This is the gap we fill.

**Bernasconi, Martino, Vittori, Trovò, and Restelli (2022)** published "Dark-Pool Smart Order Routing: a Combinatorial Multi-Armed Bandit Approach" at ICAIF 2022 (the ACM International Conference on AI in Finance). They frame dark pool SOR as a Combinatorial MAB (CMAB) problem with censored feedback, introduce the DP-CMAB algorithm, and evaluate it on real market data. Their work extends Ganchev et al. to the combinatorial action space (where the agent can route to multiple pools simultaneously) but again studies a **single institutional agent**. No competing agents or collision effects are modelled.

**Nevmyvaka, Feng, and Kearns (2006)** published the first large-scale empirical application of reinforcement learning to trade execution at ICML, using 1.5 years of NASDAQ limit order data. This paper established that RL-based execution strategies can outperform conventional approaches and laid the foundation for applying learning algorithms to financial execution.

Together, these three papers establish the legitimate application of bandit/RL methods to trade execution and dark pools. The gap this thesis occupies is the multi-agent extension: what happens when multiple learning institutions compete simultaneously, and how does the mechanism design of the pool (the collision policy) shape the collective outcome?

---

**Key papers:**

- Ganchev, K., Kearns, M., Nevmyvaka, Y., & Vaughan, J. W. (2010). Censored exploration and the dark pool problem. *Communications of the ACM*, 53(5), 109–116. (Conference version: UAI 2009.) [[CACM](https://dl.acm.org/doi/10.1145/1735223.1735247)] [[arXiv](https://arxiv.org/abs/1205.2646)]
- Bernasconi, M., Martino, S., Vittori, E., Trovò, F., & Restelli, M. (2022). Dark-pool smart order routing: a combinatorial multi-armed bandit approach. *Proceedings of the Third ACM International Conference on AI in Finance (ICAIF '22)*. [[ACM](https://dl.acm.org/doi/abs/10.1145/3533271.3561728)]
- Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for optimized trade execution. *Proceedings of the 23rd International Conference on Machine Learning (ICML 2006)*. [[ACM](https://dl.acm.org/doi/10.1145/1143844.1143929)] [[PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf)]

---

### 3.4 Dark liquidity pool microstructure

Understanding the economics of dark pools is essential for correctly interpreting the experimental results.

**Zhu (2014)** published the most cited theoretical treatment of dark pools in the *Review of Financial Studies*. His central finding is counterintuitive: adding a dark pool alongside a lit exchange *improves* price discovery on the exchange under natural conditions. The mechanism is that informed traders tend to cluster on the heavy side of the market; this makes them more likely to face execution failure in the dark pool, so they migrate to the lit market, concentrating price-relevant information there. The implication for our model is that the adverse-selection concern (which motivates `zero_on_collision`) is most acute precisely when informed and directional flow is present — i.e., exactly the scenario where institutions route large orders.

**Foley and Putniņš (2016)** tested these predictions empirically using natural experiments from dark trading restrictions in Canada and Australia (published in the *Journal of Financial Economics* as "Should We Be Afraid of the Dark?"). They find that dark limit order markets reduce spreads and improve efficiency, but the effects are heterogeneous across venue type. This heterogeneity is exactly what the three collision policies in our model are designed to capture.

**Buti, Rindi, and Werner (2011 / 2022)** provide extensive empirical analysis of dark pool trading data and identify the key determinants of dark pool activity: large stocks, high volume, low spreads, high depth. They find that dark pool trading tends to improve spreads in normal conditions. Their "Diving Into Dark Pools" paper is the primary empirical reference for understanding which stocks and market conditions see the most dark pool activity — i.e., the conditions under which our model is most relevant.

**Iyer, Johari, and Moallemi (2015)** study the welfare analysis of dark pools theoretically, examining a market with asymmetric information where traders choose between lit and dark venues. They find that poorly-designed dark pools can reduce total welfare by attracting uninformed traders and pushing informed traders to the lit market, distorting price discovery. This directly motivates the welfare analysis in RQ1: the collision mechanism is a design choice with welfare consequences.

**Almgren and Chriss (2000/2001)** provide the foundational framework for institutional trade execution, modelling the tradeoff between market impact and timing risk. The optimal execution problem they formalise is the single-agent problem that smart order routing extends to the multi-venue, multi-agent setting.

On the regulatory side, **MiFID II** (2018) represents the most significant structural change to European dark pool markets, imposing volume caps and reporting requirements that directly incentivise the fragmented, venue-diversified routing strategies that our simulations suggest emerge naturally from adaptive algorithms.

---

**Key papers:**

- Zhu, H. (2014). Do dark pools harm price discovery? *The Review of Financial Studies*, 27(3), 747–789. [[Oxford](https://academic.oup.com/rfs/article-abstract/27/3/747/1580317)]
- Foley, S., & Putniņš, T. J. (2016). Should we be afraid of the dark? Dark trading and market quality. *Journal of Financial Economics*, 122(3), 456–481. [[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X16301453)]
- Buti, S., Rindi, B., & Werner, I. M. (2011). Diving into dark pools. *Working Paper* (Fisher College of Business, Ohio State University). Published as: Dark pool trading strategies, market quality and welfare. *Journal of Financial Economics*, 124(2), 244–265 (2017). [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1630499)]
- Iyer, K., Johari, R., & Moallemi, C. C. (2015). Welfare analysis of dark pools. Working paper. [[PDF](https://moallemi.com/ciamac/papers/dark-pool-2014.pdf)]
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5–39. [[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=53501)]

---

## 4. Position of This Work in the Literature

The table below maps the most directly relevant prior work to the dimensions of this framework:

| Paper | Multiple agents | Collision model | Dark pool context | Adaptive learning |
|---|:---:|:---:|:---:|:---:|
| Ganchev et al. (2010) | ✗ | ✗ | ✓ | ✓ |
| Bernasconi et al. (2022) | ✗ | ✗ | ✓ | ✓ |
| Nevmyvaka et al. (2006) | ✗ | ✗ | ✗ | ✓ |
| Besson & Kaufmann (2018) | ✓ | ✓ | ✗ | ✓ |
| Boursier & Perchet (2019) | ✓ | ✓ | ✗ | ✓ |
| Bistritz & Leshem (2018) | ✓ | ✓ | ✗ | ✓ |
| **This framework** | **✓** | **✓** | **✓** | **✓** |

No prior work combines all four dimensions. The multi-player bandit literature studies the collision problem rigorously but without any connection to financial markets. The dark pool bandit literature (Ganchev, Bernasconi) studies the financial application but omits multi-agent competition. This framework bridges the two.

### Open questions for RQ2 extension

The communication aspect of RQ2 — which was flagged as underspecified — deserves careful treatment before implementation. Communication in multi-agent dark pool routing could take several distinct forms, each with a different empirical interpretation:

| Communication model | Dark pool interpretation |
|---|---|
| Observable actions (each agent sees others' last choice) | Post-trade transparency reports showing venue-level flow |
| Shared reward estimates | Information-sharing agreement between institutions (potentially illegal under market regulations) |
| Noisy reward signals | Second-hand market data (e.g., fill rate statistics from a data vendor) |
| Central coordinator | Coordinated trading agreement (regulated; common in some block trading contexts) |
| Adversarial / lying agents | Institutions deliberately obscuring their routing to prevent front-running |

Each maps to a different real regulatory and market scenario. The recommendation is to start with *observable actions with noise* — each agent observes a noisy version of others' last choices — as this is the most empirically grounded (post-trade reporting under MiFID II provides some, imperfect, venue flow information) and has the richest set of theoretical analogies in the multi-agent bandit literature.

---

*All literature search conducted March 2026 using Google Scholar, Semantic Scholar, and ACM Digital Library.*
