# Citation coverage audit of the skripsie (WWTW CNN classification)

Scope: every .tex file listed in the brief was read: Chapter1-6, conclusion, appendices, frontmatter, thesis.tex and mybib.bib. I also checked the compiled thesis.bbl and thesis.blg in /home/janes/Repos/Skripsie/Report. Line numbers are given as file:line. The bibliography style is `IEEEtranN` with `natbib [square,numbers,compress,sort]` (imports.tex:396-397), so citations come out numeric.

Existing entries were not checked one by one. All new DOIs were checked against the Crossref API or the arXiv API. All documentation URLs were checked with curl and returned HTTP 200, except where a note says otherwise.

---

## Q1. Which sentences make uncited claims that need a source?

### Takeaway
Chapter 1 has no citations at all. Chapter 5 cites only Optuna and pytorch-grad-cam, but it describes a full PyTorch/torchvision training stack: Adam, StepLR, early stopping, TPE, mixed precision, weighted cross-entropy and augmentation. The most serious gap is Ch5:180, which mentions "Adam's original paper" without citing it. A marker will notice that immediately. VIA, PyTorch, torchvision, TPE and the Google Earth attribution are also must-adds. Chapter 2 is the next priority: it names batch norm, layer norm, depthwise convolution / MBConv, Vision Transformers, transfer learning and the evaluation metrics without sources.

### Cited Findings

#### MUST ADD (a marker would expect these; most are one-line fixes)

1. **Ch5:180, Adam.** The sentence is: "Adam's $\beta_2$ ($0.999$, the default from Adam's original paper)". It names the paper but does not cite it, and Adam is also used throughout (Ch5:31, 228). The source is Kingma and Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015 — [arXiv:1412.6980](https://doi.org/10.48550/arXiv.1412.6980). The DOI resolves to arxiv.org/abs/1412.6980, and the title was confirmed through the arXiv API.

2. **Ch5:91 / Ch5:55 / Ch5:77 / Ch4:155, PyTorch and torchvision.**
   - The sentences are: "The augmentations are applied by PyTorch each time an image is loaded"; "All four are taken from the torchvision library"; "torchvision's `IMAGENET1K_V1` weights"; and "`torchvision.datasets.ImageFolder`".
   - PyTorch: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library", NeurIPS 2019 — [arXiv:1912.01703](https://doi.org/10.48550/arXiv.1912.01703). Verified through the arXiv API and DOI resolution.
   - torchvision: the project asks to be cited with the BibTeX key `torchvision2016`, title "TorchVision: PyTorch's Computer Vision library", author "TorchVision maintainers and contributors", 2016, howpublished GitHub — [github.com/pytorch/vision](https://github.com/pytorch/vision) (README / CITATION.cff, reported via [PyPI torchvision page](https://pypi.org/project/torchvision)).
   - Ch2:224 and the Ch5:63 table caption say parameter counts are "as defined in torchvision". Back these with the [torchvision models documentation](https://docs.pytorch.org/vision/stable/models.html), which lists the weights and parameter counts. The same page also covers the ImageNet mean/std normalisation claim at Ch5:83.

3. **Ch4:61, VGG Image Annotator.** The sentence is: "Images were manually annotated using the VGG Image Annotator (VIA2) tool." The VIA website explicitly asks users to cite Dutta and Zisserman, "The VIA Annotation Software for Images, Audio and Video", ACM Multimedia 2019 — [doi:10.1145/3343031.3350535](https://doi.org/10.1145/3343031.3350535). The DOI was confirmed through Crossref, and the request is on the [VIA site](https://www.robots.ox.ac.uk/~vgg/software/via/), which also gives ready-made BibTeX (`dutta2019vgg`).

4. **Ch5:185, TPE sampler.** The sentence is: "Optuna searches all six jointly with a Tree-structured Parzen Estimator (TPE) sampler". TPE is a named algorithm from Bergstra, Bardenet, Bengio and Kégl, "Algorithms for Hyper-Parameter Optimization", NeurIPS 24 (2011), pp. 2546-2554 — [NeurIPS proceedings PDF](https://proceedings.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf). It has no DOI. `akiba2019optuna` covers Optuna itself but not TPE.

5. **Google Earth imagery attribution** (Ch1:15, Ch1:23, Ch4:4, and all photographic figures in Ch2, Ch4 and Ch6). See Q3. A must-add, but as figure attribution, not a bib entry.

6. **Ch4:17, Northern Works figure.** The sentence is: "It is the largest wastewater treatment plant in Johannesburg and ranks among the largest in South Africa, treating approximately 400 million litres of wastewater every day."
   - [Knight Piésold, Northern Wastewater Treatment Works project page](https://knightpiesold.com/en/projects/northern-wastewater-treatment-works/) says: "The Northern WWTW is the largest of six wastewater treatment works situated in the city of Johannesburg ... Designed to treat 400 million litres of wastewater every day for approximately 1.6 million people."
   - This is a **design** capacity. The thesis says "treating", so it should be reworded to "designed to treat". Reported inflow has differed: [Mail & Guardian 2016](https://mg.co.za/article/2016-11-03-00-sewage-spill-turns-jukskei-black/) gives 460 Ml/day.
   - I found no source for "among the largest in South Africa". Drop that clause or source it separately.

7. **Ch1 motivation (whole problem statement, Ch1:4-6).** It has no citations. The claims below need a source or an attribution:
   - Ch1:4: "While facilities equipped with sensors connected to automated control systems exist globally, they are currently not widely implemented in South Africa."
   - Ch1:4: "These personnel are required to sample the water ... at least once a day in an on-site laboratory, and at least once a month, samples should be sent to an independent laboratory".
   - Ch1:6: "one or both of these testing protocols do not always occur. Incorrect measurements may be recorded, or tests may be omitted entirely."
   - Ch1:6: "on-site sensors ... are prone to failure and expensive to install and maintain."

   Sources usable here:
   - The Green Drop report that is already cited (`dws2022greendrop`) supports the data and compliance problem. Cite it again in Ch1.
   - Graham, Pattinson and Still (2025), "The state of wastewater management in South Africa: data gaps, missing wastewater, and Green Drop reporting", *Water SA* 51(2) — [doi:10.17159/wsa/2025.v51.i2.4152](https://doi.org/10.17159/wsa/2025.v51.i2.4152), open access. The DOI was confirmed through Crossref. Its abstract says Green Drop reporting was halted between 2013 and 2022, that "serious data gaps" remain, and it calls for improved monitoring ([Water SA article page](https://watersa.net/article/view/22061)). This is good open-access support for Ch1:6.
   - The `kretzmann2022` entry is currently uncited ("Half of SA's sewage treatment works are failing"). It can be cited in Ch1 for general motivation. That also removes one `\nocite` orphan.
   - The daily / monthly sampling rule: see Gaps.

8. **Ch2:84, transfer learning (defined in bold, and it is the core method).** The sentence is: "This project instead relies on **transfer learning** ...". Only ImageNet is cited. Add Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014 — [arXiv:1411.1792](https://arxiv.org/abs/1411.1792) (title confirmed via the arXiv API). Alternatively, reuse `nogueira2017` or `borzooei2024`, which are already in the bib.

9. **Ch2:249-269 and Ch1:17, evaluation metrics.** Accuracy, per-class P/R/F1, macro averaging, the confusion matrix, ROC, AUC and one-vs-rest AUC are all defined without a source.
   - Sokolova and Lapalme, "A systematic analysis of performance measures for classification tasks", *Information Processing & Management* 45(4), 2009 — [doi:10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002) (Crossref confirmed). It covers macro and micro averaging of P/R/F1.
   - Fawcett, "An introduction to ROC analysis", *Pattern Recognition Letters* 27(8), 2006 — [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010) (Crossref confirmed). It covers ROC, AUC, the AUC = 0.5 random baseline, multi-class one-vs-rest AUC, and averaging ROC curves (relevant to the fold averaging at Ch5:383).

10. **Ch5:284, mixed precision and GPU non-determinism.** The sentence is: "some GPU operations are not deterministic and mixed precision adds small rounding differences". Mixed precision is used but never introduced anywhere else.
    - Micikevicius et al., "Mixed Precision Training", ICLR 2018 — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) (arXiv API confirmed).
    - For the non-determinism claim: the [PyTorch Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) (HTTP 200).
    - For the AMP implementation: the [torch.amp docs](https://docs.pytorch.org/docs/stable/amp.html).

#### SHOULD ADD (a careful marker may comment; each is cheap)

11. **Ch2:132, Ch2:190, Ch2:210 Table, batch normalisation.** The sentences are: "$H_l$ is a composite function including batch normalisation" and "Batch normalisation is replaced with **layer normalisation**". Neither has a source.
    - Ioffe and Szegedy 2015 — [arXiv:1502.03167](https://arxiv.org/abs/1502.03167).
    - Ba, Kiros and Hinton, "Layer Normalization", 2016 — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
    - Both titles were confirmed through the arXiv API.

12. **Ch2:156-158, depthwise convolution and MBConv.** The sentence is: "EfficientNet-B0's building block is the **mobile inverted bottleneck convolution** (MBConv)". The cost formula and the "inverted bottleneck" idea come from MobileNets (Howard et al. 2017 — [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)) and MobileNetV2 (Sandler et al., CVPR 2018 — [doi:10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474), Crossref confirmed; also [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)). The k^2·C_in·HW + C_in·C_out·HW cost comparison at Ch2:158 is the MobileNets derivation, so cite Howard 2017 there.

13. **Ch2:188, Ch2:190, Ch2:197, Vision Transformer and Transformer.** The sentences are: "borrowing several ideas from Vision Transformers"; "the MLP block of a Transformer"; and "the patch-embedding step at the start of a Vision Transformer". ViT is named three times without a source. Cite Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021 — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) (arXiv API confirmed). The "MLP four times wider" point is attributed to `liu2022convnext`, which is acceptable.

14. **Ch5:89, data augmentation (whole section).** No source is given. Cite Shorten and Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning", *Journal of Big Data* 6:60, 2019, open access — [doi:10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0) (Crossref confirmed). For the exact colour-jitter semantics at Ch5:96, see the [torchvision ColorJitter docs](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html).

15. **Ch5:238, early stopping on development loss.** Cite Prechelt, "Early Stopping - But When?", in *Neural Networks: Tricks of the Trade*, LNCS 1524, 1998 — [doi:10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3) (Crossref confirmed).

16. **Ch5:124, weighted CrossEntropyLoss normalisation.** The sentence is: "`CrossEntropyLoss` ... divides the sum over a batch by the sum of these weights rather than by the number of images". This is a specific implementation claim. The [PyTorch CrossEntropyLoss docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) state that with `reduction='mean'`, "the weighted mean of the output is taken". Cite the docs page.

17. **Ch5:236, StepLR; Ch5:185, MedianPruner.** These are software features, so official docs are enough:
    - [StepLR docs](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
    - [Optuna MedianPruner docs](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html)
    - [Optuna TPESampler docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
    - The median stopping rule itself originates in Golovin et al., "Google Vizier", KDD 2017 — [doi:10.1145/3097983.3098043](https://doi.org/10.1145/3097983.3098043) (Crossref confirmed). That one is optional.

18. **Ch2:4-6, activated sludge, FBDA and clarifier description.** No source is given, and these are textbook process facts. The standard open-ish reference is Chen, van Loosdrecht, Ekama and Brdjanovic (eds.), *Biological Wastewater Treatment: Principles, Modeling and Design*, 2nd ed., IWA Publishing, 2020 — [doi:10.2166/9781789060362](https://doi.org/10.2166/9781789060362). The DOI was confirmed through Crossref, but iwaponline.com returned HTTP 403 to my client. **I could not confirm that it is open access.** A search found only paid/library listings ([British Library record](https://eld.bl.uk/catalog/019859925)). Metcalf & Eddy is the usual alternative and is not open.

19. **Ch2:8 and Ch4:63-78, visual indicators ("leopard print", black versus green clarifier surface).** The sentence is: "A healthy FBDA unit churns the surface into a 'leopard print' pattern ...". I found no published source for these satellite cues. They seem to come from the annotators and supervisor (Ch4:221 names Prof F. de Wet). Attribute them explicitly, for example "according to the annotation guidelines supplied by ..." or as personal communication, rather than leaving them as unsourced general fact. See Gaps.

#### NICE TO HAVE (low priority; textbook material)

20. **Ch2:63-82, ReLU, pooling, softmax, cross-entropy, backpropagation, vanishing gradients.** One textbook citation covers the lot: Goodfellow, Bengio and Courville, *Deep Learning*, MIT Press, 2016 — [deeplearningbook.org](https://www.deeplearningbook.org/), free online. For ReLU specifically: Nair and Hinton, ICML 2010 — [PDF](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf).

21. **Ch5:79, dropout** ("EfficientNet-B0 keeps the dropout layer"). Cite Srivastava et al., JMLR 15, 2014 — [jmlr.org](https://jmlr.org/papers/v15/srivastava14a.html). It is only mentioned in passing, so this is optional.

22. **Ch4:197-207, leave-one-facility-out / grouped splits.** `kattenborn2022` already covers this in Ch3. Optionally add Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", *Ecography* 40(8), 2017 — [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881) (Crossref confirmed). It directly supports both the blocked (leave-one-group-out) split and the chronological split at Ch4:217.

23. **Ch4:163, Python Imaging Library; Ch4:221, Jupyter widgets.** Docs links are enough if desired: [Pillow docs](https://pillow.readthedocs.io/en/stable/) and [ipywidgets docs](https://ipywidgets.readthedocs.io/en/stable/). Optional.

24. **Ch4:17, Waterval operated by ERWAT.** ERWAT runs 19 wastewater care works ([DevelopmentAid ERWAT profile](https://www.developmentaid.org/organizations/view/330611/erwat-east-rand-water-care-company)), but I found no official page that names Waterval and states its details. Leave it uncited or cite ERWAT's site; it is low risk.

25. **scikit-learn.** The thesis never says how the metrics were computed. If scikit-learn was used, which is likely for ROC/AUC, add Pedregosa et al., JMLR 2011. I did not verify a link for it, because the thesis does not name the library.

#### Uncited claims that are fine without a source (student's own work)
- Ch4 dataset counts, splits and relabelling.
- Ch5 thresholds, class-weight formula and fold averaging.
- All of Ch6 and the appendices.
These are the student's own data and results, and need no citations. Ch6 correctly has none.

### Inferences
- About 6-8 new bib entries would cover all must-adds: Kingma, Paszke, torchvision, Dutta, Bergstra, Sokolova, Fawcett, Micikevicius. About 8 more would cover the should-adds.
- Chapter 1 needs at least 2-3 citations: Green Drop, Graham 2025 or Kretzmann, plus an attribution for the sampling-frequency claim.

### Gaps
- **Ch1:4 sampling frequencies** ("at least once a day ... at least once a month ... independent laboratory"). I found no public national document that states these frequencies. A search suggests they sit in each works' water use licence under the National Water Act rather than in one national rule ([DWS Water Services Regulation chapter](https://www.dws.gov.za/Projects/National%20State%20of%20Water%20Report/Documents/9%20Water%20Services%20Regulation.pdf), not opened in full). The student should either cite the source their supervisor used or attribute it as personal communication.
- **Ch1:4 "sensors ... not widely implemented in South Africa"** and **Ch1:6 "sensors prone to failure and expensive"**: I found no direct source. Soften the wording or attribute it.
- I could not confirm whether the IWA *Biological Wastewater Treatment* book is open access.
- I found no source for Ch4:17 "ranks among the largest in South Africa", or for the Atlantis, Cape Flats and Fisantekraal location details (Ch4:17).

---

## Q2. Which existing citations are unnecessary, over-used, or wrong?

### Takeaway
Over-citation is not a real problem; only a few repeats are redundant. The bigger issue is **three citations that do not support, or contradict, the sentence they are attached to**. Fix those first.

### Cited Findings

**Citation accuracy problems (fix these):**

1. **Ch2:89, ResNet and vanishing gradients (misattribution).**
   - The thesis says ResNet addresses "the **vanishing gradient** problem" and that deep plain networks fail "purely because they become harder to optimise", with `\citep{he2016resnet}`.
   - He et al. say the opposite about the cause. The introduction states that vanishing/exploding gradients "has been largely addressed by normalized initialization ... and intermediate normalization layers". Section 4.1 says: "We argue that this optimization difficulty is unlikely to be caused by vanishing gradients. These plain networks are trained with BN ... So neither forward nor backward signals vanish." ([He et al., arXiv:1512.03385](https://arxiv.org/abs/1512.03385); text checked in the downloaded PDF.)
   - Recommendation: reword to the **degradation problem**, which is the paper's own term, and keep the citation.

2. **Ch2:84, ImageNet description.** The sentence "a benchmark dataset of over a million labelled natural images spanning a thousand object classes" is cited to `deng2009imagenet`. The 1000-class, ~1.2M-image set is the ILSVRC subset described by Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge", *IJCV* 115, 2015 — [doi:10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y) (Crossref confirmed). Deng et al. 2009 ([doi:10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)) describe the full ImageNet hierarchy, not the 1000-class benchmark. Add Russakovsky, or cite both. torchvision's `IMAGENET1K_V1` weights are trained on ImageNet-1K ([torchvision models docs](https://docs.pytorch.org/vision/stable/models.html)).

3. **Ch3:9, wrong department.** The sentence is: "The 2022 Green Drop report for the **Department of Public Works**". The bib entry, correctly, gives the Department of Water and Sanitation as author ([Graham et al. 2025](https://doi.org/10.17159/wsa/2025.v51.i2.4152) also attributes Green Drop to DWS). This is a factual error in the text.
   - Related: the sentence "Out of 115 systems, 102 are in a critical state" does not match the national headline figures that were widely reported for the 2022 report: about 850 systems, of which 334 (39%) are critical, according to search summaries of [GroundUp](https://groundup.news/article/half-of-south-africas-sewage-treatment-works-are-failing/) and [Polity](https://www.polity.org.za/article/national-blue-green-drop-and-no-water-watch-reports-released---urgent-intervention-required-2023-06-07). I did not open those pages myself.
   - If "115" is a provincial or subset figure, the text must say which. Whoever is verifying `dws2022greendrop` should check this.

**Redundant or over-used (low priority):**
- Ch3:37 re-cites all four architectures `\citep{he2016resnet, huang2017densenet, tan2019efficientnet, liu2022convnext}` after they were already cited in Ch2:84. This is harmless and could be dropped, since Ch2 is the place where they are introduced. Self-evident from /home/janes/Repos/Skripsie/Report/chapters/Chapter3.tex:37.
- `he2016resnet` appears at Ch2:84, 89 and 190, plus Ch3:37 and Ch5:57. The Ch2:190 instance supports a specific number (256→64→256), so it is justified. This is not a problem.
- `eoportal_quickbird` at Ch3:25 supports a parenthetical about QuickBird. It is acceptable but marginal, and could be dropped if space is tight.
- Ch5:180 uses `\cite{akiba2019optuna}` where everything else uses `\citep`. With numeric natbib the output is identical (imports.tex:396), so this is cosmetic only.

**Missed opportunities with existing entries (these remove `\nocite` orphans):**
- Ch3:23 says "fine-tuned a pretrained AlexNet" with no citation for AlexNet. Cite the existing, currently uncited `krizhevsky2012alexnet` here.
- Ch1 motivation: cite the existing `kretzmann2022` entry (see Q1 item 7).
- `westbrook2016` ("Havana's wastewater treatment plants: changes over time ...") fits Ch3:30, on remote sensing of treatment plants, if it uses historical imagery. I did not read it. If it does not fit, delete it.
- Ch3:14 says "Transfer learning from ImageNet generally offers better performance than training from scratch." It has no citation, and it reads like a general claim. Attach `\citep{borzooei2024}` (if it is that paper's finding) or `nogueira2017`.

### Inferences
- Of all the issues in this file, the ResNet misattribution (item 1) is the one a computer-vision marker is most likely to flag. It is a one-word fix: replace "vanishing gradient" with "degradation".

### Gaps
- I did not verify the "59%" or the "115/102" figures in the Green Drop report itself. The Western Cape mirror PDF ([link](https://d7.westerncape.gov.za/eadp/files/atoms/files/Report_NATIONAL-_FINAL_30March22_MNEdit_web.pdf)) returned an HTML page instead of a PDF to my client.

---

## Q3. Imagery licensing and attribution (Google Earth)

### Takeaway
**Must add.** Google's terms require on-image attribution that names Google **and** the third-party imagery providers. That attribution must stay with each image, not be gathered at the end. Every satellite-image figure in the thesis needs it. A citation in the bibliography alone does not satisfy the terms.

### Cited Findings
- Google's Geo Guidelines require attribution in every use. The attribution must stay "with the content", must not be moved "to the end of your book", must be legible, and must name the providers. Naming only Google is not enough: "only including 'Google' or the Google logo is not proper attribution." — [Google Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/)
- For print uses within the guidelines, "you don't need to submit a request". Books are allowed up to 5k copies, and reports/business documents and periodicals are allowed. Google Earth content may be used for research and education with attribution, and not for commercial purposes. Google says it "can't tell you if your use of this content would be fair use". — [Google Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/)
- Affected figures, all built from Google Earth crops: Ch2:15-27 (`aerobicfunc_background`, `aerobicdysfunc_background`, `clarifier_background`); Ch4:9 heatmap (data only, fine); Ch4:21 map (check what basemap was used); Ch4:29-53 facility overviews; Ch4:106 and Ch4:112 class examples; Ch4:228 and Ch4:235 relabel figures; Ch5:107 augmentation; Ch6 Grad-CAM figures. Source: /home/janes/Repos/Skripsie/Report/chapters/*.tex.

### Inferences
- Practical fix: keep the original Google Earth attribution strip in each overview image, or add "Imagery: Google Earth, © [year] [provider, e.g. Maxar Technologies / Airbus / CNES]" to each figure caption. For the small crops, one sentence in Ch4:4 plus a caption note on each figure is defensible.
- The provider names and years should be read off the Google Earth Pro attribution of each historical image. They vary by date, and I cannot determine them from here.
- An acknowledgement of Google Earth in the front matter would be a good addition, but it does not replace per-figure attribution under these terms.

### Gaps
- I did not determine the actual providers for each facility and date; the student must read them in Google Earth Pro.
- Google Earth Pro also has its own [Google Earth Additional Terms of Service](https://support.google.com/earth/answer/21417?hl=en) page (the URL returned HTTP 200, but I did not read its contents).

---

## Q4. Do the Stellenbosch template entries need to be cited?

### Takeaway
No. Remove `\nocite{*}` (thesis.tex:71) and they drop out of the reference list automatically. Neither entry supports any claim in the text.

### Cited Findings
- `sun_thesis_guide` is "Stellenbosch University, *Formatting your thesis*, 2024, bit.ly link". `sun_latex_template` is "Danie Els, *Stellenbosch University thesis bundle*, 2024, CTAN". Both are leftover template entries, not sources (/home/janes/Repos/Skripsie/Report/mybib.bib:1-12).
- A shortened `bit.ly` URL in a reference list is poor practice in any case (mybib.bib:5).
- Under IEEEtran sentence casing, the template title renders as "Stellenbosch **university** thesis bundle" (thesis.bbl:136).

### Inferences
- If the student wants to credit the template, a line in the acknowledgements is the normal place. The acknowledgements still contain template placeholder text: "Supervisor X", "Funding Y", "Thank you to Leanne for helping to prepare this template" (frontmatter/acknowledgements.tex:6-17).
- Other leftover template text outside the citation scope, worth fixing before submission:
  - thesis.tex:23 still says "PhD (Electronic Engineering)".
  - The acronym list has "MFT, my funky thesis" (frontmatter/defined_acronyms.tex).
  - Nomenclature placeholders remain (frontmatter/variables_and_functions.tex).
  - The declaration date is "10 May 2024" (frontmatter/declaration.tex).
  - The abstract is empty (frontmatter/abstract.tex).

### Gaps
- None.

---

## Q5. Bibliography formatting consistency

### Takeaway
The entries are mostly well formed and brace-protected. However, **none of the 22 DOIs in the bib appear in the PDF**, because `IEEEtranN.bst` v1.14 ignores the `doi` field. Several proper nouns are also lowercased by IEEEtran's sentence-casing. The DOI problem has a two-minute fix.

### Cited Findings
- **The DOIs are invisible.** thesis.bbl is "Generated by IEEEtranN.bst, version: 1.14 (2015/08/26)", and `grep -c doi thesis.bbl` returns 0, even though 22 entries carry a `doi` field. For example, `borzooei2024` renders without its DOI (thesis.bbl:57-58). Fix it in one of two ways:
  - add `url = {https://doi.org/...}` to each entry (IEEEtran prints `url`); or
  - switch to a DOI-aware style.
  Source: /home/janes/Repos/Skripsie/Report/thesis.bbl.
- **Sentence-case damage** (IEEEtran lowercases titles of `@article`, `@inproceedings`, `@techreport` and `@misc`). Wrap these words in braces:
  - `dws2022greendrop` renders as "Green **drop** national report 2022". Use `{Green Drop}`.
  - `szegedy2016inception` renders as "Rethinking the **inception** architecture". Use `{Inception}`.
  - `sun_latex_template` renders as "Stellenbosch **university** ...". This entry should be deleted anyway.
  - `kretzmann2022` renders as "```a tsunami of human waste'". The opening backtick in the title gets merged and lowercased. Use `{\textquoteleft}A tsunami...` or drop the inner quotes.
  - Possibly also check "Belgian" (none in titles) and `costa2013` "a review" (fine).
  Source: thesis.bbl:52, 124, 136, 142.
- **Brace protection is otherwise good.** `{ImageNet}`, `{CNNs}`, `{EfficientNet}`, `{ConvNet}`, `{Grad-CAM}`, `{Optuna}`, `{GELUs}`, `{Google Earth}`, `{OpenStreetMap}`, `{OSM}`, `{ROC}`, `{PatternNet}`, `{QuickBird-2}` and `{Clever Hans}` are all protected (mybib.bib).
- **Missing DOIs.** All the older ML and CNN entries lack a DOI or URL, while the Ch3 entries have them, which is inconsistent:
  - `lecun1998gradient` has a verified DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791).
  - `he2016resnet` has a verified DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
  - `deng2009imagenet` has a verified DOI: [10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848).
  - `selvaraju2017gradcam` has a verified DOI: [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74).
  - The other CNN entries (huang2017densenet, hu2018squeeze, tan2019efficientnet, liu2022convnext, szegedy2016inception, akiba2019optuna, krizhevsky2012alexnet) also lack DOIs or URLs. I did not look these up, because the other researchers own the per-entry checks.
- **Entry-type inconsistencies:**
  - `hendrycks2016gelu` is an `@article` with journal "arXiv preprint arXiv:1606.08415". This is acceptable for IEEEtran, but `@misc` with `eprint`/`url` would be cleaner.
  - `geirhos2019` puts its arXiv URL in `note` rather than citing the ICLR OpenReview page.
  - Online sources (`gildenblat2021pytorchcam`, `eoportal_quickbird`, `kretzmann2022`, `westbrook2016`, `dws2022greendrop`) put `\url{}` and "Accessed ..." in `note`. This works, but only some have access dates: `gildenblat2021pytorchcam` and `dws2022greendrop` do not.
  - `dws2022greendrop` links to the IRIS results portal (`ws.dws.gov.za/IRIS/latestresults.aspx`), not to the report itself.
  - `lapuschkin2019` has no issue number.
  - `eoportal_quickbird` gives year 2025 but was accessed in 2026.
  Source: /home/janes/Repos/Skripsie/Report/mybib.bib.
- **Author formatting.** Initial-only authors (`costa2013`, `satoh2021`) sit alongside full-name authors elsewhere. IEEEtran abbreviates first names anyway, so the output is consistent.
- **The `\nocite{*}` problem** is already known (thesis.tex:71). Removing it is the correct fix, after first citing `krizhevsky2012alexnet` and `kretzmann2022` as suggested in Q2.
- `thesis.blg` shows 0 BibTeX warnings, so there are no structurally missing required fields.

### Inferences
- Priority order for the bib:
  1. Make the DOIs render.
  2. Remove `\nocite{*}`.
  3. Add braces to {Green Drop} and {Inception}, and fix the Kretzmann quote.
  4. Add DOIs to the CNN entries for consistency.

### Gaps
- I did not check each existing entry's metadata (pages, volumes, author lists), as the brief instructed; other researchers own that.
