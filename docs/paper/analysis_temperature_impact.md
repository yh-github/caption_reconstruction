# Analysis of Temperature Impact on Reconstruction Quality

This document analyzes the impact of sampling temperature on the reconstruction quality of the `phi-3` model, specifically investigating why high temperature (T=1.5) occasionally outperforms lower temperatures on specific metrics like Recall@1, despite generally poorer performance.

## Case Study 1: Weathershot_7-clip-0

**Video Context**: A VR game (Beat Saber style) involving dodging/hitting colored blocks.

### File Links
- **Original Data**: [Weathershot_7-clip-0.json](../../datasets/wildQA/captions__wild2/Weathershot_7-clip-0.json)
- **T=0.1 Reconstruction**: [phi-3__t=0.1_rp=1.2 (w=12, i=29)](../../results/recon/wild_dev_sim_text/reconstruction/wild_dev_sim_text/phi-3__t=0.1_rp=1.2__fixed_fill(w=12, i=29)/Weathershot_7-clip-0.json)
- **T=1.5 Reconstruction**: [phi-3__t=1.5_rp=1.2 (w=12, i=29)](../../results/recon/wild_dev_sim_text/reconstruction/wild_dev_sim_text/phi-3__t=1.5_rp=1.2__fixed_fill(w=12, i=29)/Weathershot_7-clip-0.json)

### Comparison

| Metric | T=0.1 Score | T=1.5 Score | Winner |
| :--- | :--- | :--- | :--- |
| **Recall@1** | Low (Generic) | **High** | **T=1.5** |
| **Qualitative** | High (Accurate) | Low (Hallucination) | T=0.1 |

#### Caption Samples

**Original (Ground Truth)**
> "...ducks and dodges **colorful virtual blocks** that fly towards them..."

**T=0.1 (Low Temperature - Accurate)**
> "Mask 24: At this moment, our protagonist skillfully maneuvers between rapidly descending **colored spheres** within a dynamic obstacle course..."
> "Mask 29: Our hero confidently leaps aside just in time to avoid colliding with speedily approaching **multicolor drones**..."

*Observation*: The T=0.1 reconstruction is semantically very close to the original. "Colored spheres" is a reasonable approximation of "colorful virtual blocks".

**T=1.5 (High Temperature - Hallucinated)**
> "Mask 24: In masterfully crafted prose befitting said task... verily 'Amidst triumphantly collectible puzzle pieces..."
> "Mask 25: Amid his successful collection journey by gathering **tantric jigsaw components**..."
> "Mask 33: Catch all those flying colored bits smooth! Keep scoring points galore! Stay **windbustzy**!!!"
> "Mask 35: Throughout thirty hours since conception - HHMMSTTTZGLLXSITTLMCTTPBXXPSLRQFBBYJNFTLFXCWTMTLSNTLTKHHILGGTYOULNGGLCPTFEYHTOTELSEICPXT"

*Observation*: The T=1.5 reconstruction generates incoherent, high-entropy text ("tantric jigsaw", "windbustzy") and even crashes into character soup at the end. It completely fails to describe the video accurately.

### Analysis
Why does T=1.5 "win" on Recall@1?
- **Uniqueness vs. Accuracy**: The Recall@1 metric measures if the reconstructed text is closest to the *original video's clip index* in a retrieval task.
- **The "Generic" Trap**: T=0.1 produces "colored spheres". Many other VR games or sci-fi videos might also have "colored spheres". This makes the embedding for T=0.1 somewhat generic, potentially retrieving the wrong clip (or being retrieved by the wrong query).
- **The "Hallucination" Edge**: T=1.5 produces unique tokens like "tantric jigsaw" and "Northern Ghurans" (in other examples). These are extremely rare. If the model happens to associate this specific "noise" uniquely with this clip idx (perhaps due to overfitting or random chance in the high-dim space), it scores a "hit".
- **Conclusion**: The "win" is an artifact of the metric and the high variance of the output, not a sign of quality.

---

## Case Study 2: Army-military-2018_8-clip-73

**Video Context**: A large explosion on a battlefield.

### File Links
- **Original Data**: [Army-military-2018_8-clip-73.json](../../datasets/wildQA/captions__wild2/Army-military-2018_8-clip-73.json)
- **T=0.1 Reconstruction**: [phi-3__t=0.1_rp=1.2 (w=12, i=29)](../../results/recon/wild_dev_sim_text/reconstruction/wild_dev_sim_text/phi-3__t=0.1_rp=1.2__fixed_fill(w=12, i=29)/Army-military-2018_8-clip-73.json)
- **T=1.5 Reconstruction**: [phi-3__t=1.5_rp=1.2 (w=12, i=29)](../../results/recon/wild_dev_sim_text/reconstruction/wild_dev_sim_text/phi-3__t=1.5_rp=1.2__fixed_fill(w=12, i=29)/Army-military-2018_8-clip-73.json)

### Comparison

#### Caption Samples

**Original (Ground Truth)**
> "Index 2: A massive fireball erupts from the grove of trees..."
> "Index 3: The explosion expands, sending a large cloud of orange fire..."

**T=0.1 (Low Temperature - Accurate)**
> "Mask 26: ...our protagonist stands motionless beside what appears now only partially intact—a once formidable barrier..."
> "Mask 27: In silence broken suddenly; chaos unfolds behind him - he feels vibrations..."

*Observation*: Captures the tension and the event ("chaos unfolds", "barrier").

**T=1.5 (High Temperature - Hallucinated)**
> "Mask 48: **Chad discovers evidence** pointing toward unseen consequences... conflict between **Northern Ghurans** militant groups..."
> "Mask 50: At dawn following multiple devastations incited primarily via subterranean displeasances..."

*Observation*: Completely hallucinates a character ("Chad") and a specific geopolitical conflict ("Northern Ghurans") that are not present in the video visually.

### Conclusion
Similar to the Weathershot example, T=1.5 generates specific proper nouns and entities ("Chad", "Ghurans") which likely aid in unique retrieval (Recall@1) despite being factually wrong relative to the visual content.
