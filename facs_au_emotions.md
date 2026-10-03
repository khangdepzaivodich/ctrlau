# EMFACS & Compound Emotion AU Rules (Verified)

This document specifies the canonical relationship between **Facial Action Units (AUs)** and **Emotions** based on:
1. **FACS & EMFACS (Emotion Facial Action Coding System)**: Ekman & Friesen (1978); Ekman, Friesen, & Hager (2002); Matsumoto & Ekman (2008).
2. **FACSAID (FACS Affect Interpretation Dictionary)**: Ekman, Friesen, & Hager (2002).
3. **Compound Facial Expressions of Emotion**: Du, Tao, & Martinez (*PNAS*, 2014).

---

## 1. Universal Prototypical EMFACS AU-Emotion Rules (Full FACS)

| Emotion | Required Core AUs | Typical / Common Variations | Notes & Biomechanics |
| :--- | :--- | :--- | :--- |
| **Happiness** | **AU 6 + AU 12** | AU 25, AU 26 | **Duchenne smile**: Cheek raiser (*Orbicularis oculi*) + Lip corner puller (*Zygomaticus major*). AU 25/26 appear in open-mouth smiles or laughter. |
| **Sadness** | **AU 1 + AU 4 + AU 15** *(or AU 1 + AU 4 + AU 17)* | AU 11, AU 17, AU 25, AU 26 | Inner brow raiser (AU 1) + Brow lowerer (AU 4) form the characteristic "sadness brow". Lip corner depressor (AU 15) and/or Chin raiser (AU 17) anchor the lower face. |
| **Surprise** | **AU 1 + AU 2 + AU 5 + (AU 26 OR AU 27)** | AU 25 | Inner (AU 1) & Outer (AU 2) brow raisers pull the whole eyebrow up, Upper lid raiser (AU 5) exposes sclera above iris, and Jaw drops (AU 26) or stretches (AU 27). Per FACS 2002 additive standard, AU 26/27 includes AU 25. |
| **Fear** | **AU 1 + AU 2 + AU 4 + AU 5 + AU 20** | AU 7, AU 25, AU 26, AU 27 | Full fear brow (AU 1+2 raised while AU 4 pulls medially) + eye widening (AU 5) + Lip stretcher (AU 20). Lower face typically parts lips (AU 25) or drops jaw (AU 26/27). AU 7 (lid tightener) is a frequent co-activator. |
| **Anger** | **AU 4 + AU 5 + AU 7 + (AU 23 OR AU 24)** | AU 17, AU 22, AU 25, AU 26, AU 10 | Eyebrows pulled down and together (AU 4), eyes widened with glare (AU 5) or narrowed/tightened (AU 7). Lips are tightened (AU 23) or pressed shut (AU 24). In shouting/aggressive anger, AU 10, AU 25, AU 26 may occur. |
| **Disgust** | **AU 9 OR AU 10** | AU 15, AU 16, AU 17, AU 25, AU 26, AU 4 | **Core marker**: Nose Wrinkler (AU 9, olfactory disgust) **or** Upper Lip Raiser (AU 10, oral/taste disgust), or both (AU 9 + AU 10). Accompanied optionally by lower lip depression (AU 16), chin raise (AU 17), lip corner depression (AU 15), or parting lips (AU 25/26). *(Note: AU 15 + AU 16 are NOT simultaneously mandatory).* |
| **Contempt** | **Unilateral AU 14** *(or unilateral AU 12)* | AU 25 | Characterized by asymmetry: unilateral dimpler (AU 14, *Buccinator*) tightening one lip corner, or unilateral lip corner puller (AU 12). |

---

## 2. Compound Emotion AU Rules (Du, Tao, & Martinez, PNAS 2014)

Compound emotions combine Action Units from two basic emotions into distinct, visually consistent patterns:

| Compound Emotion | Component Basic Emotions | Core Distinguishing AUs |
| :--- | :--- | :--- |
| **Happily Surprised** | Happiness + Surprise | **AU 1 + AU 2 + AU 5 + AU 12 + AU 25** *(± AU 26)* |
| **Happily Disgusted** | Happiness + Disgust | **AU 6 + AU 12 + (AU 9 OR AU 10) + AU 25** |
| **Angrily Surprised** | Anger + Surprise | **AU 4 + AU 5 + AU 25 + AU 26** *(± AU 1, AU 2)* |
| **Fearfully Surprised** | Fear + Surprise | **AU 1 + AU 2 + AU 5 + AU 20 + AU 25** *(± AU 26)* |
| **Sadly Fearful** | Sadness + Fear | **AU 1 + AU 4 + AU 20 + AU 25** |
| **Sadly Angry** | Sadness + Anger | **AU 1 + AU 4 + AU 7 + AU 15 + AU 17** |
| **Sadly Disgusted** | Sadness + Disgust | **AU 1 + AU 4 + (AU 9 OR AU 10) + AU 15** |
| **Fearfully Angry** | Fear + Anger | **AU 4 + AU 5 + AU 20 + AU 25** |
| **Disgustedly Surprised** | Disgust + Surprise | **AU 1 + AU 2 + AU 5 + (AU 9 OR AU 10) + AU 25** |

---

## 3. Adaptation to the DISFA Benchmark Subset (8 AUs)

Benchmark facial expression datasets like **DISFA** only annotate an 8-AU subset:
$$\text{DISFA AUs} = \{\text{AU 1}, \text{AU 2}, \text{AU 4}, \text{AU 6}, \text{AU 9}, \text{AU 12}, \text{AU 25}, \text{AU 26}\}$$

Because key units (AU 5, 7, 10, 14, 15, 17, 20, 23, 24) are unannotated in DISFA, training losses (such as `FACSEmotionViolationLoss` and graph routing in `ctrlau`) project the full EMFACS definitions onto the available observable space:

| Emotion | Full FACS Observable in DISFA | Projected Active Rule in `config.py` | Rationale & Missing Units |
| :--- | :--- | :--- | :--- |
| **Happiness** | AU 6, AU 12, AU 25, AU 26 | **AU 6 AND AU 12** | Full Duchenne smile preserved completely in DISFA. |
| **Sadness** | AU 1, AU 4 | **AU 1 AND AU 4** | AU 15 and AU 17 are unannotated in DISFA; sadness brow (AU 1 + AU 4) serves as the primary observable proxy. |
| **Surprise** | AU 1, AU 2, AU 25, AU 26 | **AU 1 AND AU 2 AND AU 26** | AU 5 unannotated in DISFA; upper brow raise (AU 1+2) + jaw drop (AU 26) unambiguously identifies surprise. |
| **Fear** | AU 1, AU 2, AU 4, AU 25, AU 26 | **AU 1 AND AU 2 AND AU 4 AND AU 26** | AU 5 and AU 20 unannotated in DISFA; fear brow (AU 1+2+4) + jaw drop (AU 26) proxy. |
| **Anger** | AU 4 | **AU 4** | AU 5, 7, 17, 23, 24 unannotated in DISFA; brow lowerer (AU 4) is the sole available marker. |
| **Disgust** | AU 9, AU 25, AU 26 | **AU 9** | AU 10 and AU 15 unannotated in DISFA; Nose Wrinkler (AU 9) is the primary diagnostic landmark for olfactory disgust. |

---

## 4. Key Corrections vs. Common Misconceptions

1. **Disgust does NOT require `AU 9 + AU 15 + AU 16`**:
   - `AU 9` alone or `AU 10` alone is sufficient for prototypical disgust under EMFACS.
   - AU 15 and AU 16 are lower face modifiers that occur in specific sub-varieties, not invariant requirements.
2. **Additive Mouth Opening (`AU 25` with `AU 26/27`)**:
   - In modern FACS (2002), jaw drop (`AU 26`) or stretch (`AU 27`) physically forces lips to part (`AU 25`). Coders score both (`AU 25 + AU 26`). They are not mutually exclusive.
3. **Fear Mouth Variations**:
   - The defining fear mouth is `AU 20` (Lip Stretcher), which can combine with `AU 25` (parted lips), `AU 26` (dropped jaw), or `AU 27` (mouth stretch).
