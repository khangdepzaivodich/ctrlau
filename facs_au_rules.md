# FACS Action Unit Semantic & Anatomical Rules

This document specifies the anatomical, kinematic, and semantic relationships between Facial Action Units (AUs) based on the **Facial Action Coding System (FACS 2002 Manual)** by Ekman, Friesen, & Hager, verified against affective computing literature (e.g., Tong et al., IEEE TPAMI 2007; Cohn et al., 2007).

---

## 1. Valid Mutually Exclusive Rules (XOR)

These actions involve mutually exclusive physical muscle configurations, bilateral vs. unilateral definitions, or opposite kinematic rotational directions for rigid / conjugate structures:

### A. Facial Muscle Contraction Incompatibilities
* **AU 24 XOR AU 25** (Lip Pressor $\oplus$ Lips Part)
  * *Anatomy*: AU 24 tightens and presses the red margins of the lips together (*Orbicularis oris* contraction). AU 25 separates the lips. The lips cannot be simultaneously pressed shut and parted at the same location.

### B. Eyelid State Alternatives
* **AU 41 XOR AU 42 XOR AU 43** (Lid Droop $\oplus$ Slit $\oplus$ Eyes Closed)
  * *Anatomy*: Represents mutually exclusive degrees along an ordinal eyelid closure scale for a given eye.
* **AU 45 XOR AU 46** (Blink $\oplus$ Wink)
  * *Anatomy*: AU 45 is defined as a rapid, bilateral eye closure; AU 46 is defined as a deliberate unilateral eye closure. A single eye-closure event cannot be scored as both simultaneously.

### C. Rigid Head Kinematics (Opposing Rotations)
* **AU 51 XOR AU 52** (Head Turn Left $\oplus$ Head Turn Right — Yaw)
* **AU 53 XOR AU 54** (Head Pitch Up $\oplus$ Head Pitch Down — Pitch)
* **AU 55 XOR AU 56** (Head Tilt Left $\oplus$ Head Tilt Right — Roll)
  * *Kinematics*: The skull is a rigid body; it cannot rotate in opposite directions along the same rotational axis at the same time instance.

### D. Conjugate Eye Movements (Opposing Gaze Directions)
* **AU 61 XOR AU 62** (Eyes Turn Left $\oplus$ Eyes Turn Right — Horizontal Gaze)
* **AU 63 XOR AU 64** (Eyes Up $\oplus$ Eyes Down — Vertical Gaze)
  * *Physiology*: Normal conjugate gaze shifts eyes in a single directional vector at any instant.

---

## 2. Corrected Hierarchy: Mouth Opening (Additive, NOT XOR)

> ⚠️ **FACS 1978 Shorthand vs. FACS 2002 Additive Standard**
>
> * **Incorrect 1978 Rule**: `AU 25 XOR AU 26 XOR AU 27` or `AU 27 subsumes AU 25 and AU 26`.
>   * In the 1978 manual, coders used a shorthand where higher mouth opening omitted lower codes to save notation time.
> * **Modern FACS 2002 Standard**: **Additive Coding**.
>   * Dropping the jaw (**AU 26**) or stretching the mouth open (**AU 27**) anatomically causes the lips to part (**AU 25**).
>   * Therefore, in benchmark datasets (e.g., DISFA, BP4D, CK+), jaw drop is coded as **AU 25 + AU 26**, exhibiting **>70% empirical co-occurrence**.
>   * **Correct Implication Rule**:
>     $$\text{AU 26} \implies \text{AU 25} \quad (p(\text{AU 25}) \ge p(\text{AU 26}))$$
>     $$\text{AU 27} \implies \text{AU 25} \quad (p(\text{AU 25}) \ge p(\text{AU 27}))$$

---

## 3. Dominance & Masking Rules (Corrected from "Subsuming")

In standard FACS terminology, Action Units are independent and do not "subsume" (eliminate) one another. However, strong muscle actions can visually **mask** or cause **passive secondary movement**:

### A. AU 9 (Nose Wrinkler) vs. AU 4 (Brow Lowerer)
* **Rule**: AU 9 does **NOT** subsume AU 4.
* *Anatomical Fact*: *Levator labii superioris alaeque nasi* (AU 9) inserts near the medial brow and can cause slight passive tension at the inner brow.
* *FACS Coding Criterion*: If the subject actively contracts the *Corrugator supercilii*, both **AU 4 and AU 9** are scored together (frequent in disgust and pain). If the brow lowering is solely passive skin pull from intense AU 9, AU 4 is not scored.

### B. AU 9 (Nose Wrinkler) vs. AU 10 (Upper Lip Raiser)
* **Rule**: AU 9 can **mask** AU 10 (NOT "AU 10 subsumes AU 9").
* *Anatomical Fact*: AU 9 raises both the nose root and the center of the upper lip. Strong AU 9 makes the distinct contribution of AU 10 (*Levator labii superioris*) difficult to discern. Coders only score AU 10 in the presence of AU 9 if lateral lip pull indicates additional muscle recruitment.

### C. AU 6 (Cheek Raiser) vs. AU 7 (Lid Tightener)
* **Rule**: Intense AU 6 (*Orbicularis oculi, pars orbitalis*) pushes up the lower eyelid, which can mask AU 7 (*Orbicularis oculi, pars palpebralis*).

---

## 4. Application to the DISFA 8-AU Benchmark Subset

The standard DISFA evaluation set comprises 8 Action Units:
$$\text{DISFA AUs} = \{\text{AU 1}, \text{AU 2}, \text{AU 4}, \text{AU 6}, \text{AU 9}, \text{AU 12}, \text{AU 25}, \text{AU 26}\}$$

* **Mutual Exclusivity within DISFA**:
  * AU 24, AU 27, AU 41–46, and head/eye units (51–64) are **not present** in the 8-AU benchmark set.
  * AU 25 and AU 26 are frequently co-active (additive smile / open-mouth speech). Enforcing an `AU 25 XOR AU 26` penalty introduces substantial false training loss.
* **Co-occurrence / Synergistic Pairs in DISFA**:
  * **Happiness / Smile**: AU 6 + AU 12
  * **Surprise / Fear Brow Raise**: AU 1 + AU 2
  * **Mouth Opening / Speech**: AU 25 + AU 26
  * **Disgust / Pain Expression**: AU 4 + AU 9

---

## 5. AU-to-Expression (AU-Exp) Prior Rules (EMFACS)

For the complete, scientifically verified AU-to-Emotion mappings (including basic emotions under EMFACS and 9 compound emotions from Du, Tao, & Martinez, PNAS 2014), refer to [`facs_au_emotions.md`](facs_au_emotions.md).

### Summary of Canonical AU-Exp Prototypical Rules:
* **Happiness**: AU 6 + AU 12 (Duchenne smile; optional AU 25, 26).
* **Sadness**: AU 1 + AU 4 + AU 15 (or AU 1 + AU 4 + AU 17; optional AU 11, 25, 26).
* **Surprise**: AU 1 + AU 2 + AU 5 + (AU 26 OR AU 27) (optional AU 25).
* **Fear**: AU 1 + AU 2 + AU 4 + AU 5 + AU 20 (optional AU 7, 25, 26, 27).
* **Anger**: AU 4 + AU 5 + AU 7 + (AU 23 OR AU 24) (optional AU 17, 22, 25, 26, 10).
* **Disgust**: AU 9 OR AU 10 (core diagnostic units; optional AU 15, 16, 17, 25, 26). Note: AU 15 and 16 are NOT simultaneously required.
* **Contempt**: Unilateral AU 14 (Dimpler) or unilateral AU 12.

