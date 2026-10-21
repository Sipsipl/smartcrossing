# Literature and audiences (research notes, 2026-10-14)

Quick web research (abstracts, project pages, one full read of Pal et al. via ar5iv). Not a systematic
review; details marked "unverified" come from secondary summaries.

## Closest prior work

- **Pal, Philion, Liao, Fidler: "Emergent Road Rules in Multi-Agent Driving Environments", ICLR 2021**
  ([arXiv 2011.10753](https://arxiv.org/abs/2011.10753), code: fidler-lab/social-driving).
  2D environments (synthetic 4-way crossing, highway, nuScenes patches). Shared PPO policy, CTDE,
  reward +1 goal, -1 collision/off-road (no blame), plus jerk and distance-to-goal shaping. Steering is
  NOT free: a sub-policy picks a spline path, another the acceleration. Perception = LiDAR rays; noise =
  dropping 25/50/75 % of rays per step; density = number of agents. Emerged: stopping at (given)
  traffic signals, lanes (more with noise and density; low density -> roundabout-like motion), right of
  way (~85 % FIFO), fast lane, minimum distance, slowing near crosswalks. One-bit turn-signal channel
  (only from the car ahead): its use grows with perception noise. Deadlocks / negotiation: not studied.
- **Rui Zhang (HKUST Physics), talk "A Traffic-Rule Multiverse Emerges from Multi-Agent RL"**,
  Academia Sinica, 2026-08-05 ([announcement](https://www.phys.sinica.edu.tw/lecture_detail.php?id=3231)).
  Abstract: agents on two-lane roads, cross intersections, roundabouts develop ranked passing priority,
  keep-left/right, first-come-first-go; reward / parameter sharing change the "character". No paper
  found yet. Very close to our questions -> potential contact.
- **Nakao, Kasaura, Kozuno: "Symmetry-Breaking in Multi-Agent Navigation" (arXiv 2511.15239, 2025).**
  Decentralised agents without shared intentions fall into symmetry-induced deadlocks; they learn a
  topological passing strategy (winding number) + MPC. Engineered solution, not emergent conventions.
- **Grupen, Lee, Selman: "Multi-Agent Curricula and Emergent Implicit Signaling", AAMAS 2022**
  ([arXiv 2106.11156](https://arxiv.org/abs/2106.11156)): implicit signalling through motion (pursuit-
  evasion), measured via social influence / bits exchanged. Relevant method for analysing our lamp and
  "body language".
- Engineering MARL for unsignalised intersections (many 2024-2026 papers, e.g. attention + game priors
  [arXiv 2409.05712](https://arxiv.org/abs/2409.05712), right-of-way MARL, centralised coordinators):
  mostly impose right-of-way rules, V2V state sharing or central coordination; no free steering, no
  emergent negotiation. Surveys note missing deadlock-freedom guarantees for learned policies.
- Conventions as equilibrium selection / symmetry breaking (Sen & Airiau 2007 precedence rules on grids;
  ad hoc teamwork symmetry-breaking augmentations, arXiv 2402.09984; Zanlungo 2008 evolved flow
  conventions) - background for (G).

## Where Smart Crossing differs (candidate contributions)

1. Free 2D steering without lanes or paths (Pal: spline paths; most others: fixed lanes).
2. A free, meaning-less lamp channel visible to all neighbours (Pal: 1 bit from the car ahead only;
   many: explicit V2V state sharing).
3. Destinations hidden from others (nothing to leak), so any lamp meaning must be invented.
4. Deadlocks and their decentralised resolution as an explicit research question (G) - not studied by
   Pal; engineered by Nakao et al.
5. Many seeds cheaply (GPU sim, ~500k car-steps/s) - most emergence papers report few seeds.
6. Live in-browser simulation of every brain (rare; strong for communication and review).
Caveat: novelty must be re-checked before writing (citing papers of Pal et al., Zhang's work).

## Audiences and formats (realistic for a solo author without affiliation)

- Interactive explainer (explorable, distill-style) based on the site: no review, widest reach.
- JOSS (Journal of Open Source Software): peer-reviewed short paper about the environment as research
  software; no novelty requirement, needs docs/tests/usage examples. Realistic.
- ALIFE 2027 (Prague, 19-23 July 2027; deadlines likely ~March/April 2027, unverified): emergence-friendly
  community; full papers or summaries.
- Workshops on multi-agent learning / cooperative AI / emergent communication (NeurIPS, ICML, AAMAS
  extended abstracts): 4-8 pages; check the 2027 calls.
- arXiv preprint alongside (first submission needs an endorsement).
- Contacts: Rui Zhang (HKUST), the Fidler lab / Avik Pal, Nakao et al. - a short email with a concrete
  result and the live demo.
