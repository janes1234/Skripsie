# Fix the sentences, not the sources

## 1. Verdict

**Every one of the 41 entries in `mybib.bib` is a real source, and none looks fabricated.** Authors, titles and venues match Crossref, arXiv, PMLR, NeurIPS or the publisher page in every case. The bibliography is not the weak point. The weak point is a handful of sentences that say something the cited source does not say. The worst is Chapter2.tex:89, which credits ResNet with solving the vanishing gradient problem. The ResNet paper says the opposite. Four more are close behind. Chapter3.tex:23 says a working clarifier should look *brighter*, which contradicts the cited review and the thesis's own Chapter2.tex:8 and Chapter6.tex:176. Chapter3.tex:9 uses Green Drop figures for prison and state plants, and gets the direction of the paperwork gap backwards. Chapter3.tex:44 turns Kattenborn's "up to 28%" into a flat 28%. Chapter3.tex:25 gives QuickBird sub-metre resolution in four bands. There are also mechanical problems a marker will see straight away. **None of the 22 DOIs prints**, because IEEEtranN ignores the `doi` field. `\nocite{*}` pads the list with five uncited entries, two of them template leftovers. The Green Drop link now opens the 2025 report. Three entries carry page numbers that do not match their DOI. Chapter 1 has no citations at all. Chapter 5 names "Adam's original paper" without citing it, and never cites PyTorch, torchvision, VIA or TPE. No satellite figure carries the Google Earth attribution that Google's terms require. My overall rating of the referencing is **6/10**. The sources are good and the coverage of Chapter 3 is good. The 6 comes from five misused sentences, the invisible links, and a methods chapter that cites almost none of the tools it describes. Every fix below is small. The bibliography-only fixes take about an hour. The text fixes are a few sentences each.

Two earlier researcher claims are wrong, so do not act on them. The sentence at Chapter5.tex:57 is not a leftover from commit 732e95b ("removed inception and resnet 50"). It deliberately says ResNet-50 and Inception-v3 were evaluated but not reported for space reasons, which is exactly what the commit did. Keep it, and keep `szegedy2016inception` and `he2016resnet` cited there. Chapter3.tex:9 also does not name the "wrong department". The national Green Drop report has a chapter on the Department of Public Works' own systems, and the 115/102 figures come from it ([DWS national report PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf)). The problem with that sentence is what the numbers are used to argue, not the department named.

## 2. Every cited source, checked

The links below were all verified by the researchers: DOIs resolved and matched to Crossref or DataCite, arXiv IDs confirmed through the arXiv API, and other pages fetched with HTTP 200. "Checked against" shows how deep the claim check went. "Metadata only" means the existence and the bibliographic details were confirmed, but the cited sentence was not compared with the text of the source.

| Key | Cited at | Verdict | Verified link | Claim checked against |
|---|---|---|---|---|
| lecun1998gradient | Ch2:37 | OK. Add `url`. | [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) | Metadata only (standard CNN citation, low risk) |
| deng2009imagenet | Ch2:84, Ch5:77 | Real, but does not support "over a million images, a thousand classes" | [doi:10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848), [open PDF](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf) | Full text |
| he2016resnet | Ch2:84, 89, 190; Ch3:37; Ch5:57 | Real. **Ch2:89 contradicted by the paper.** Other uses OK. | [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90), [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | Full text |
| huang2017densenet | Ch2:84, 127; Ch3:37 | Claims OK. **Pages wrong for DOI** (IEEE 2261–2269). | [doi:10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243), [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) | Full text |
| szegedy2016inception | Ch5:57 | OK. Title prints "inception" in lower case. | [doi:10.1109/CVPR.2016.308](https://doi.org/10.1109/CVPR.2016.308) | Metadata only |
| hu2018squeeze | Ch2:156 | OK | [doi:10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745), [arXiv:1709.01507](https://arxiv.org/abs/1709.01507) | Full text |
| tan2019efficientnet | Ch2:84, 156; Ch3:37 | OK. No DOI exists; use the PMLR page. | [PMLR v97](https://proceedings.mlr.press/v97/tan19a.html), [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) | Full text |
| liu2022convnext | Ch2:84, 188, 190; Ch3:37 | Claims OK. **Pages wrong for DOI** (IEEE 11966–11976). | [doi:10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167), [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | Full text |
| akiba2019optuna | Ch5:180 | OK | [doi:10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701) | Metadata only (it is the Optuna paper) |
| selvaraju2017gradcam | Ch2:276; Ch3:46 | OK | [doi:10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74), [arXiv:1610.02391](https://arxiv.org/abs/1610.02391) | Full text |
| gildenblat2021pytorchcam | Ch5:404 | OK. Matches the README's requested citation. | [GitHub](https://github.com/jacobgil/pytorch-grad-cam) | README, plus installed source (see below) |
| hendrycks2016gelu | Ch2:195 | OK. arXiv only; no peer-reviewed version. | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | Full text |
| dws2022greendrop | Ch3:9 | Real. **Bib URL now opens the 2025 report.** **Claim direction reversed.** | [2022 national PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf), [Wayback copy](https://web.archive.org/web/20220710190541/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf) | Full text (558 pages) |
| costa2013 | Ch3:14 | Real. Missing issue number and DOI. | [doi:10.1007/s11356-013-1824-5](https://doi.org/10.1007/s11356-013-1824-5) | Search snippet only (weakest check in the list) |
| satoh2021 | Ch3:14 | OK | [doi:10.1039/D0EW00908C](https://doi.org/10.1039/D0EW00908C) | Abstract |
| borzooei2024 | Ch3:14, 18, 35, 37 | OK. One small overstatement at Ch3:14. | [doi:10.1016/j.jwpe.2024.105692](https://doi.org/10.1016/j.jwpe.2024.105692), [arXiv v3](https://arxiv.org/abs/2402.09367v3) | Full text (preprint v3) |
| carballomato2024 | Ch3:16, 18, 37 | OK | [doi:10.3390/w16030390](https://doi.org/10.3390/w16030390) | Full text |
| gholizadeh2016 | Ch3:23, 25 | Partly supported. **The sentence after it contradicts it.** | [doi:10.3390/s16081298](https://doi.org/10.3390/s16081298), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/) | Full text |
| lopezbetancur2022 | Ch3:23 | OK | [doi:10.3390/app12126079](https://doi.org/10.3390/app12126079) | Full text |
| lesiv2018 | Ch3:25 | OK | [doi:10.3390/land7040118](https://doi.org/10.3390/land7040118) | Abstract |
| eoportal_quickbird | Ch3:25 | Page real. **Sentence wrong on resolution.** Publisher "ESA" unconfirmed. | [eoPortal](https://www.eoportal.org/satellite-missions/quickbird-2) (alternative: [ESA Earth Online](https://earth.esa.int/eogateway/missions/quickbird-2)) | Full page |
| hu2013 | Ch3:25 | OK | [doi:10.3390/rs5116026](https://doi.org/10.3390/rs5116026) | Abstract |
| zhou2018patternnet | Ch3:30 | OK | [doi:10.1016/j.isprsjprs.2018.01.004](https://doi.org/10.1016/j.isprsjprs.2018.01.004), [arXiv:1706.03424](https://arxiv.org/abs/1706.03424) | Full text |
| li2022wwtp | Ch3:30 | OK | [doi:10.1016/j.jag.2022.102804](https://doi.org/10.1016/j.jag.2022.102804), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC9626640/) | Full text |
| shuai2023 | Ch3:30 | OK | [doi:10.3390/su15075880](https://doi.org/10.3390/su15075880) | Full text |
| randhawa2023 | Ch3:30 | OK | [doi:10.1145/3615900.3628772](https://doi.org/10.1145/3615900.3628772) | Abstract |
| handannader2019 | Ch3:30 | OK. Every number matches the abstract. | [doi:10.1038/s41893-019-0246-x](https://doi.org/10.1038/s41893-019-0246-x) | Abstract (paywalled) |
| nogueira2017 | Ch3:35 | OK | [doi:10.1016/j.patcog.2016.07.001](https://doi.org/10.1016/j.patcog.2016.07.001), [arXiv:1602.01517](https://arxiv.org/abs/1602.01517) | Full text |
| penatti2015 | Ch3:35 | Fact OK. The inference drawn from it is stretched. | [doi:10.1109/CVPRW.2015.7301382](https://doi.org/10.1109/CVPRW.2015.7301382), [CVF open PDF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Penatti_Do_Deep_Features_2015_CVPR_paper.pdf) | Full text |
| geirhos2019 | Ch3:37 | OK. ICLR; no DOI. | [arXiv:1811.12231](https://arxiv.org/abs/1811.12231) | Full text |
| buda2018 | Ch3:39, 48 | OK. Wrong verb at Ch3:39 ("developed"). | [doi:10.1016/j.neunet.2018.07.011](https://doi.org/10.1016/j.neunet.2018.07.011), [arXiv:1710.05381](https://arxiv.org/abs/1710.05381) | Full text |
| lin2017focal | Ch3:39 | Claim OK. **Pages wrong for DOI** (IEEE 2999–3007). | [doi:10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324), [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | Full text |
| kattenborn2022 | Ch3:44 | Real. **Number overstated** ("up to 28%" written as "28%"). | [doi:10.1016/j.ophoto.2022.100018](https://doi.org/10.1016/j.ophoto.2022.100018) (open), [KIT copy](https://publikationen.bibliothek.kit.edu/1000156879) | Full text |
| geirhos2020 | Ch3:46 | OK | [doi:10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z), [arXiv:2004.07780](https://arxiv.org/abs/2004.07780) | Full text |
| lapuschkin2019 | Ch3:46 | OK. The paper says "source tag", not "copyright tag". | [doi:10.1038/s41467-019-08987-4](https://doi.org/10.1038/s41467-019-08987-4) | Full text |
| saito2015 | Ch3:48 | OK | [doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432) | Title (the claim restates it) |
| *Not cited:* sun_thesis_guide, sun_latex_template, krizhevsky2012alexnet, kretzmann2022, westbrook2016 | reach the list only through `\nocite{*}` | See Section 5 | | |

One gap from the notes is now closed. The researchers could not confirm that pytorch-grad-cam normalises each heatmap on its own, which is what Chapter5.tex:404 claims. The installed version (`grad_cam` 1.5.7 in `/home/janes/Repos/Skripsie/.venv`) has `scale_cam_image` in `pytorch_grad_cam/utils/image.py`, and that function min-max scales each CAM in a loop over images. The sentence is correct.

## 3. Where the text does not match the source

These are ordered from most to least serious. Suggested wording keeps the existing sentence wherever possible.

### 3.1 Chapter2.tex:89: ResNet is credited with solving vanishing gradients, which the paper denies (severity: high)

The thesis says ResNet addresses "the vanishing gradient problem". He et al. say vanishing gradients had "been largely addressed by normalized initialization ... and intermediate normalization layers". They also write: "We argue that this optimization difficulty is unlikely to be caused by vanishing gradients ... We also verify that the backward propagated gradients exhibit healthy norms with BN." The problem they name is the **degradation problem** ([He et al., arXiv:1512.03385](https://arxiv.org/abs/1512.03385)). A computer-vision marker who knows the paper will catch this. The second sentence, about plain networks training worse with depth, is supported. Chapter2.tex:91 ("a gradient ... always has a direct, unimpeded path") is a common textbook explanation. It can stay, but it is not the paper's own argument.

Suggested replacement for the first two sentences:

> Residual Networks (ResNet) \citep{he2016resnet} address a specific problem for CNNs, which the authors call the \textbf{degradation} problem. Beyond a certain depth, plain networks made only of stacked convolutions begin to perform worse, even on the training set, purely because they become harder to optimise, not because they lack capacity. The authors argue that this is unlikely to be caused by vanishing gradients, since their plain networks use batch normalisation and their gradients keep healthy norms.

If `ioffe2015batchnorm` is added (Section 6), cite it after "batch normalisation".

### 3.2 Chapter3.tex:23: "should look brighter" has the physics backwards (severity: high)

Gholizadeh et al. say "Suspended sediments are responsible for most of the scattering" and that "TSM increases with increasing reflectance" ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/)). A clarifier that removes solids should therefore reflect *less* light and look darker, not brighter. The thesis already says this twice: Chapter2.tex:8 says a working clarifier "looks black from above", and Chapter6.tex:176 contrasts a misclassified "pale grey" surface with the "darker green" of a correct one. A marker reading Chapter 3 next to Chapters 2 and 6 will see the contradiction. The first sentence also overstates the review. Gholizadeh covers chlorophyll-a, Secchi depth, CDOM, phosphorus, temperature and salinity as well, so satellite studies do not "focus on" suspended solids and turbidity.

> Satellite studies of water quality often estimate suspended solids and turbidity, because water clarity falls as the suspended solids concentration rises \citep{gholizadeh2016}. Suspended solids cause most of the scattering of light in water \citep{gholizadeh2016}, so a clarifier that eliminates solids should look darker than one that retains them.

### 3.3 Chapter3.tex:9: the Green Drop figures argue the opposite of what the paragraph claims (severity: high)

The numbers are real. The report says "one hundred and two (102 of 115) systems were identified in critical state in the DPW" (p. 451, and p. xi). It treats "A deviation of >10% between the GD and TSA score" as "a misalignment between the administrative aspects and the work on the ground" (p. 469). It reports DPW deviations "ranging from 21% to 59%" (p. 469) ([DWS national report PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf)). The paragraph has three problems.

The first is direction. In almost every DPW case the **site inspection score is higher than the paperwork score**. Waterval Prison scored 0% on paper and 59% on site (Table 16, pp. 468–469). So the paperwork *understated* the plant. That undermines the closing claim that paperwork "can hide true performance".

The second is relevance. DPW systems are prisons, border posts and similar state facilities, not municipal works like the plants in this dataset.

The third is precision. The 59% is a gap of 59 percentage points between two scores. It is not "59% of the real state".

Rating the paragraph as motivation in its current form: **4/10**. It is a real number from a real report, used for the wrong argument.

The better option, rated **8/10**, is to use the municipal figures from the same report. These are "334 (39%) of municipal wastewater systems ... in a critical state" (p. xi), and municipal desktop/site deviations of 0% to 57%, with KwaZulu-Natal at 56% "indicating a severe disconnect between wastewater administration and field conditions" (p. 32). One caveat: the report rates the City of Cape Town's systems well (site scores above 80%, close to their paperwork scores, p. 31). The paragraph must not suggest that the Cape Town plants in the dataset are among the failing ones.

Minimal fix that keeps the DPW numbers:

> The 2022 Green Drop report found that 102 of the 115 wastewater systems run by the Department of Public Works were in a critical state \citep[p.~451]{dws2022greendrop}. The report treats a difference of more than 10\% between the desktop audit score and the site inspection score as a misalignment between administration and the work on the ground, and found differences of up to 59 percentage points for these systems \citep[p.~469]{dws2022greendrop}. This project proposes using imagery to spot these differences, as the report indicates that scores based on submitted paperwork do not always match the condition of the plant.

Municipal alternative (recommended):

> The 2022 Green Drop report found that 334 of 850 municipal wastewater systems were in a critical state \citep[p.~xi]{dws2022greendrop}. The report treats a difference of more than 10\% between the desktop audit score and the site inspection score as a misalignment between administration and the work on the ground, and found differences of up to 57 percentage points between municipalities \citep[p.~32]{dws2022greendrop}. This project proposes using imagery to spot these differences, as the report indicates that scores based on submitted paperwork do not always match the condition of the plant.

Whichever you choose, the bib URL must change. It currently opens the 2025 report (Section 7).

### 3.4 Chapter3.tex:44: Kattenborn's worst case is reported as the result (severity: medium)

The abstract says random holdouts "overestimated the model performance by up to 28%", "assuming that a block cross-validation provides a realistic model performance". The 28% (27.9%) applies only to the smallest training set of 10 orthoimages. With 40 orthoimages it was 14.6%. The metric is the F1-score of a segmentation task ([Kattenborn et al.](https://doi.org/10.1016/j.ophoto.2022.100018); [KIT full text](https://publikationen.bibliothek.kit.edu/1000156879)). "Than it actually was" also claims more than the paper, which only *assumes* block validation is realistic. It is **uncertain** whether the 28% is a relative change or a difference in F1 points. The paper's usage suggests points. The sentence also uses the second person ("how you select", "when you only have"), which does not match the rest of the thesis.

> \citet{kattenborn2022} demonstrated that how test data are selected is critical in remote sensing. When analysing drone imagery, choosing random test images made the model's F1-score appear up to 28\% higher than with spatially separate test data. [...] To get a realistic performance score, the model must be tested on entirely separate geographic locations, a rule that becomes even more important when the amount of data is small.

### 3.5 Chapter2.tex:84 and Chapter5.tex:77: Deng 2009 does not describe the 1000-class benchmark (severity: medium)

Deng et al. (2009) describe a database of "5247 synsets and 3.2 million images" and never mention a 1000-class subset ([open PDF](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)). The 1000-class set with about 1.2 million training images, which is what `IMAGENET1K_V1` means, is ILSVRC-2012. It is described in Russakovsky et al. 2015 ([doi:10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y)). Keep Deng and add Russakovsky to both sentences: `\citep{deng2009imagenet, russakovsky2015imagenet}`. No wording change is needed.

### 3.6 Chapter3.tex:25: QuickBird's sub-metre resolution is panchromatic only (severity: medium)

The 0.61 m resolution applies to the single panchromatic band. The four multispectral bands are 2.4 m ([eoPortal](https://www.eoportal.org/satellite-missions/quickbird-2); [ESA Earth Online](https://earth.esa.int/eogateway/missions/quickbird-2)). The difference matters here, because the paragraph's argument rests on spectral bands.

> \citet{hu2013} compared Google Earth imagery for land cover mapping with imagery from QuickBird, a commercial satellite with a sub-metre panchromatic band and four multispectral bands \citep{esa_quickbird}.

Optionally add "in one case study in Wuhan" after "land cover mapping". Hu et al. is a single-city study ([doi:10.3390/rs5116026](https://doi.org/10.3390/rs5116026)).

### 3.7 Chapter4.tex:17: Northern Works capacity is uncited and is a design figure (severity: medium)

Knight Piésold's project page says the plant is "the largest of six wastewater treatment works situated in the city of Johannesburg" and was "Designed to treat 400 million litres of wastewater every day" ([Knight Piésold](https://knightpiesold.com/en/projects/northern-wastewater-treatment-works/)). That is design capacity, not throughput. Reported inflow has differed: the Mail & Guardian gave 460 Ml/day in 2016 ([Mail & Guardian](https://mg.co.za/article/2016-11-03-00-sewage-spill-turns-jukskei-black/)). No source was found for "ranks among the largest in South Africa", so drop that clause.

> It is the largest wastewater treatment plant in Johannesburg and is designed to treat approximately 400 million litres of wastewater every day \citep{knightpiesold_northern}.

A municipal (Johannesburg Water) source would be better than a contractor's page, but none was found.

### 3.8 Chapter3.tex:35: the Penatti caution goes further than the paper (severity: low to medium)

The fact is right: a colour descriptor (BIC) beat ImageNet features on the coffee scenes. But those scenes are green, red and near-infrared SPOT composites, not RGB. The authors blame "finer and more homogeneous textures" as much as colour. The gap was small, 87.03% against 84.82% ([CVF PDF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Penatti_Do_Deep_Features_2015_CVPR_paper.pdf)). Rating the inference "may miss a cue when classes differ mainly in colour, as the clarifier classes do" as written: **5/10**.

> ImageNet features gave the best results on aerial scenes but were beaten by a simple colour descriptor on multispectral (green, red and near-infrared) scenes of coffee crops, which the authors attribute partly to colour and partly to fine, uniform texture.

### 3.9 Smaller mismatches (severity: low)

**Chapter3.tex:46.** Lapuschkin et al. call the mark a "source tag", present in about a fifth of horse images. The word "copyright" does not appear in the paper ([doi:10.1038/s41467-019-08987-4](https://doi.org/10.1038/s41467-019-08987-4)). Write "by a source tag (a watermark) present in about a fifth of the horse images". Also fix "woud".

**Chapter3.tex:14 (Borzooei).** The paper says the underestimation "may be linked to a bias in the training data, which mainly includes instances of lower SVI values" ([arXiv v3](https://arxiv.org/abs/2402.09367v3)). That is weaker than "a very small number of such samples". Write "which the authors link to a training set made up mainly of low index values". The uncited general sentence "Transfer learning from ImageNet generally offers better performance than training from scratch" sits inside the Borzooei paragraph and is that paper's finding, so attribute it: "They found that transfer learning from ImageNet gave better results than training from scratch."

**Chapter3.tex:39.** Buda et al. tested oversampling; they did not develop it ([doi:10.1016/j.neunet.2018.07.011](https://doi.org/10.1016/j.neunet.2018.07.011)). Write "Both were studied on data with distinct classes". Focal loss was proposed for dense object detection ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002)), and saying so would be more exact. Separately, "Uneven class distributions adds" should be "add".

**Chapter3.tex:30 (PatternNet).** PatternNet has 800 image chips per class, so "one label per plant" is loose ([arXiv:1706.03424](https://arxiv.org/abs/1706.03424)). Write "with a single class label regardless of condition". Optionally write Shuai's figure as "80.1\% precision at 90.4\% recall". The paper calls the value accuracy, but it is computed as 137 of 170 detections ([MDPI PDF](https://mdpi-res.com/d_attachment/sustainability/sustainability-15-05880/article_deploy/sustainability-15-05880.pdf)).

**Chapter3.tex:30 and 52 ("none of these studies asks whether the tanks are working").** This is true of the detection papers, but one source does ask the question: `westbrook2016`, which the thesis has but does not cite. Westbrook and de Freitas Alves judged from historical Google Earth Pro images whether units at six Havana plants were in service, for example "the WWTP was operational, as identified by water in both circular clarifiers" ([ASCE Cuba database](https://ascecubadatabase.org/asce_proceedings/havanas-wastewater-treatment-plants-changes-over-time-and-estimate-of-replacement-cost/)). Citing it makes the novelty claim honest instead of weaker. A marker who finds this essay will think worse of an uncited "no study was found". Add after the "none of these studies" sentence:

> The only attempt found to judge from such imagery whether units were in service is a manual comparison of historical Google Earth images of six plants around Havana \citep{westbrook2016}, which inferred, for example, that a plant was operating because its clarifiers held water.

Then change Chapter3.tex:52 to "no study was found that automatically classifies the state of treatment units from satellite imagery".

**Chapter2.tex:190.** "making training less sensitive to batch size" is not ConvNeXt's stated reason. The paper only says BN "has many intricacies" ([arXiv:2201.03545](https://arxiv.org/abs/2201.03545)). It is a fair general point about layer normalisation. Add `ba2016layernorm` to the LN definition so the ConvNeXt citation is not read as supporting it. Whether Ba et al. state the batch-size point in those words is **uncertain**, because no researcher checked their text.

**Chapter3.tex:23 (Lopez-Betancur).** Optional. The samples were distilled water mixed with fish feed or paprika, made for aquaculture ([MDPI PDF](https://mdpi-res.com/d_attachment/applsci/applsci-12-06079/article_deploy/applsci-12-06079.pdf)). "Prepared liquid samples" would say so.

Two minor points need no change. At Chapter2.tex:132, DenseNet-121's H_l actually contains two convolutions (BN-ReLU-Conv1×1-BN-ReLU-Conv3×3), so "a convolution" understates it but is not false ([arXiv:1608.06993](https://arxiv.org/abs/1608.06993)). At Chapter2.tex:278, Grad-CAM "uses the output of the last convolutional layer" is the paper's recommended default rather than a rule. It is acceptable ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391)).

## 4. Hard-to-verify sources and easier alternatives

None of the 41 entries looks fabricated. "Hard to verify" here means a marker clicking through will hit a paywall, a moved page, or grey literature.

| Key | Why it is hard to verify | Recommendation |
|---|---|---|
| dws2022greendrop | The bib link (`ws.dws.gov.za/IRIS/latestresults.aspx`) now opens with "It is a great honour to present the Green Drop Report 2025" ([IRIS](https://ws.dws.gov.za/IRIS/latestresults.aspx)). **A marker cannot find the cited report from the reference list.** This is the worst findability problem in the bibliography. | Link the 2022 PDF directly ([PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf)) plus the [Wayback copy](https://web.archive.org/web/20220710190541/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf), and cite page numbers. If you keep the DPW numbers, the separate DPW sub-report also exists ([PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_DPW_Rev03_05May22_MN%20web.pdf), [Wayback](https://web.archive.org/web/20220710190925/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_DPW_Rev03_05May22_MN%20web.pdf)). The national report with page numbers is enough. |
| eoportal_quickbird | The page is real. The bib says "European Space Agency", but eoPortal's about page names no operator ([eoPortal about](https://www.eoportal.org/about)). The page also contradicts itself on the mission end date. | Switch to [ESA Earth Online](https://earth.esa.int/eogateway/missions/quickbird-2), which gives the same specifications and has an unambiguous publisher. Rating the switch: 7/10. Entry `esa_quickbird` is in the appendix. |
| costa2013 | Springer is paywalled. The listed open copy (hdl.handle.net/1822/25500) returned HTTP 500. The claim check rests on the title, the venue and a search snippet. | Keep it. The claim "has grown into a field of its own" only needs a 25-page review to exist, and Crossref confirms it does. Add the DOI. **Uncertain** beyond that. If you have library access, skim the abstract. |
| satoh2021 | Only the abstract was read. A green open-access copy is listed at Hokkaido University, but it returned HTML to the script, so that link is **unverified**. | Keep it. The DOI works and the abstract supports Ch3:14. |
| handannader2019 | Paywalled. | Keep it. Every number in Ch3:30 is in the open abstract ([Nature](https://www.nature.com/articles/s41893-019-0246-x)). |
| geirhos2019, hendrycks2016gelu, tan2019efficientnet | No DOI (ICLR, arXiv, PMLR). | Not a problem. Put the arXiv or PMLR link in `url`. The OpenReview page for Geirhos could not be verified, so use arXiv. |
| westbrook2016 | Grey literature: an undergraduate prize essay, 2016 ASCE Student Competition. It may also appear in *Cuba in Transition* vol. 26, but that is **unverified**. | Cite it, and describe it honestly as a manual study. The page is live and the entry is correct as it stands. |
| kretzmann2022 | Journalism (GroundUp, republished by Daily Maverick) that repeats DWS figures. | Delete it (Section 5). |
| sun_thesis_guide | A `bit.ly` link, which is poor practice in any reference list. | Delete it (Section 5). |
| gildenblat2021pytorchcam | GitHub only. This is the form the project asks for. | Keep it, and add the version used (1.5.7, read from the installed package metadata). |

## 5. The five uncited entries

All five appear only because of `\nocite{*}` at thesis.tex:71. Remove that line in every case. Then each entry either gets a real `\cite` or is deleted.

| Key | Decision | Reason |
|---|---|---|
| sun_thesis_guide | **Delete** | A template leftover. It supports no claim and has a bit.ly link. |
| sun_latex_template | **Delete** | A template leftover. It renders as "Stellenbosch university thesis bundle". To credit the template, thank Danie Els in the acknowledgements, which still contain placeholder text (frontmatter/acknowledgements.tex:6–17). |
| krizhevsky2012alexnet | **Keep and cite at Chapter3.tex:23** | The researchers disagreed. Reading the line settles it. Chapter3.tex:23 says "\citet{lopezbetancur2022} fine-tuned a pretrained AlexNet". The thesis cites every other architecture it names, so naming AlexNet without a citation is the odd one out. Write "a pretrained AlexNet \citep{krizhevsky2012alexnet}". Rating: **7/10**. The researcher's 3/10 "delete" assumed a new padding sentence would be needed. It is not. The NeurIPS page ([proceedings](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)) gives no page numbers, so the commonly quoted 1097–1105 is **unverified**. The appendix entry leaves pages out. |
| kretzmann2022 | **Delete** | One recommendation, as asked. Its numbers (334 of 850 critical) come from the DWS report, which is already cited and is the primary source. Its one distinctive point is that Green Point and Hout Bay hold Green Drop awards while being marine outfalls ([Daily Maverick](https://www.dailymaverick.co.za/article/2022-06-21-a-tsunami-of-human-waste-half-of-sas-sewage-treatment-works-are-failing-says-study/)). That concerns plants outside the dataset that are not activated sludge works. For Chapter 1 motivation, the DWS report and Graham et al. (2025) are stronger, so citing a news piece there adds a weaker source next to a stronger one. Rating for keeping it anywhere: **4/10**. The two researchers rated it 5/10 and "cite in Ch1"; both are too generous. |
| westbrook2016 | **Keep and cite at Chapter3.tex:30** | It is the closest precedent to the whole thesis (Section 3.9). Rating: **8/10**. It could also go at Chapter1.tex:4 ("manual analysis of aerial photography"). |

## 6. Citations to add

### Must add

A marker will expect each of these. Most are a single `\citep`.

| Where | What the sentence does | Source | Verified link |
|---|---|---|---|
| Chapter5.tex:180 | Names "Adam's original paper" without citing it | Kingma and Ba, ICLR 2015 (`kingma2015adam`) | [doi:10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980) |
| Chapter5.tex:55, 77, 91; Chapter4.tex:155 | Uses PyTorch and torchvision models, weights and `ImageFolder` | Paszke et al., NeurIPS 2019 (`paszke2019pytorch`); TorchVision maintainers (`torchvision2016`) | [doi:10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703); [github.com/pytorch/vision](https://github.com/pytorch/vision) (citation form reported via [PyPI](https://pypi.org/project/torchvision)) |
| Chapter4.tex:61 | Uses VIA2, whose authors ask to be cited | Dutta and Zisserman, ACM MM 2019 (`dutta2019vgg`) | [doi:10.1145/3343031.3350535](https://doi.org/10.1145/3343031.3350535); request on the [VIA site](https://www.robots.ox.ac.uk/~vgg/software/via/) |
| Chapter5.tex:185 | Names the TPE algorithm | Bergstra et al., NeurIPS 2011 (`bergstra2011tpe`) | [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) (no DOI; link given by the researcher, not explicitly reported as fetched) |
| Chapter2.tex:84, Chapter5.tex:77 | 1000 classes, about 1.2M images | Russakovsky et al., IJCV 2015 (`russakovsky2015imagenet`) | [doi:10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y) |
| Chapter1.tex:6 | Whole problem statement, currently uncited | `dws2022greendrop` (p. xi) and Graham, Pattinson and Still 2025, *Water SA* (`graham2025`), open access, which reports that Green Drop reporting stopped from 2013 to 2022 and that "serious data gaps" remain | [doi:10.17159/wsa/2025.v51.i2.4152](https://doi.org/10.17159/wsa/2025.v51.i2.4152), [article page](https://watersa.net/article/view/22061) |
| Chapter4.tex:17 | 400 Ml/day figure | Knight Piésold project page (`knightpiesold_northern`) | [knightpiesold.com](https://knightpiesold.com/en/projects/northern-wastewater-treatment-works/) |
| Chapter3.tex:23, Chapter3.tex:30 | Existing entries (see Section 5) | `krizhevsky2012alexnet`, `westbrook2016` | Section 2 |
| Every satellite figure | Google Earth imagery | Caption attribution, not a bib entry (Section 7) | [Google Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/) |

Suggested Chapter 1 sentence, added at the end of the paragraph at Chapter1.tex:6:

> National reporting on wastewater treatment in South Africa has serious data gaps \citep{graham2025}, and the 2022 Green Drop audit found 334 of 850 municipal systems in a critical state \citep[p.~xi]{dws2022greendrop}.

Three Chapter 1 claims could not be sourced by anyone. The first is the daily on-site and monthly independent sampling rule (Chapter1.tex:4). It probably sits in each works' water use licence rather than in one national rule, but that is **unverified**. The second is "not widely implemented in South Africa" (Chapter1.tex:4). The third is that sensors "are prone to failure and expensive" (Chapter1.tex:6). For each, either cite whatever document your supervisor relied on, or attribute it in the text (for example "according to [name], personal communication"). Do not leave them as unsourced general fact. The visual cues at Chapter2.tex:8 ("leopard print", black versus green clarifiers) have the same problem. No published source exists for them, and they appear to come from the annotation guidelines prepared under Prof F. de Wet (Chapter4.tex:221). Say so in one clause.

### Should add

A careful marker may comment on these. Each is cheap.

| Where | Source | Verified link |
|---|---|---|
| Chapter2.tex:249–269 (metrics) | Sokolova and Lapalme 2009 (macro P/R/F1); Fawcett 2006 (ROC, AUC, one-vs-rest, averaging curves) | [doi:10.1016/j.ipm.2009.03.002](https://doi.org/10.1016/j.ipm.2009.03.002); [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010) |
| Chapter2.tex:132, 190 (BN, LN) | Ioffe and Szegedy 2015; Ba, Kiros and Hinton 2016 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167); [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |
| Chapter2.tex:156–158 (depthwise cost formula, MBConv) | Howard et al. 2017 (MobileNets, the source of the cost comparison); Sandler et al. 2018 (MobileNetV2, the inverted bottleneck) | [arXiv:1704.04861](https://arxiv.org/abs/1704.04861); [doi:10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474) |
| Chapter2.tex:188, 197 (ViT, named three times) | Dosovitskiy et al., ICLR 2021 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| Chapter2.tex:84 (training from scratch needs large data) | Reuse `nogueira2017`, which found full training "was not a good strategy" for small datasets. No new entry needed. | Section 2 |
| Chapter5.tex:284 (mixed precision, GPU non-determinism) | Micikevicius et al., ICLR 2018; PyTorch Reproducibility notes | [arXiv:1710.03740](https://arxiv.org/abs/1710.03740); [PyTorch docs](https://docs.pytorch.org/docs/stable/notes/randomness.html) |
| Chapter5.tex:89 (augmentation) | Shorten and Khoshgoftaar 2019, open access | [doi:10.1186/s40537-019-0197-0](https://doi.org/10.1186/s40537-019-0197-0) |
| Chapter5.tex:238 (early stopping) | Prechelt 1998 | [doi:10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3) |
| Chapter5.tex:124 (weighted mean in `CrossEntropyLoss`) | PyTorch docs ("the weighted mean of the output is taken") | [docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| Chapter2.tex:224, Chapter5.tex:63 caption, Chapter5.tex:83 (parameter counts, ImageNet normalisation) | torchvision models documentation | [docs](https://docs.pytorch.org/vision/stable/models.html) |
| Metrics code | scikit-learn is used in `CNN/wwtw_utils.py` and `CNN/make_results_notebooks.py`, but the thesis never names it. Name it where the metrics are computed and cite Pedregosa et al., JMLR 2011. | **Link not verified by any researcher.** Check the JMLR page before adding it. |
| Chapter2.tex:4–6 (activated sludge, FBDA, clarifiers) | Chen, van Loosdrecht, Ekama and Brdjanovic (eds.), *Biological Wastewater Treatment*, 2nd ed., IWA 2020. Whether it is open access is **unconfirmed**. | [doi:10.2166/9781789060362](https://doi.org/10.2166/9781789060362) |
| Chapter2.tex:63–82 (ReLU, pooling, softmax, cross-entropy, backpropagation) | Goodfellow, Bengio and Courville, *Deep Learning*, 2016. One citation covers about six uncited concepts. Rating 6/10. | [deeplearningbook.org](https://www.deeplearningbook.org/) (link given by the researcher, not explicitly reported as fetched) |

### Skip

These add length without adding trust. Nair and Hinton for ReLU and Srivastava et al. for dropout are covered by the Goodfellow citation, and dropout is only mentioned in passing. Golovin et al. (Google Vizier) for the median stopping rule, and the Optuna `MedianPruner`, `TPESampler` and `StepLR` documentation pages, are covered by `akiba2019optuna` and `bergstra2011tpe`. Roberts et al. 2017 on grouped cross-validation ([doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)) is good but redundant with `kattenborn2022`; add it only if a marker asks. Yosinski et al. 2014 is the canonical transfer-learning paper, but reusing `nogueira2017` does the job. Pillow and ipywidgets docs, and an ERWAT source for Waterval, are low risk to leave uncited. So is `kretzmann2022`.

Re-citing the four architectures at Chapter3.tex:37 is not over-citation. It supports the "published between 2016 and 2022" claim, so keep it. `eoportal_quickbird` (or `esa_quickbird`) is marginal but justified by the QuickBird sentence.

## 7. Bibliography mechanics

**DOIs do not print.** `IEEEtranN.bst` v1.14 has no `doi` handling, and `grep -c doi thesis.bbl` returns 0, although 22 entries carry a DOI. The style does print `url` as "[Online]. Available: ...". A test compile with the thesis's own package set (hyperref, url, xurl, natbib `[square,numbers,compress,sort]`, IEEEtranN) confirmed that `url = {https://doi.org/...}` prints and is clickable. The fix is to add a `url` field holding the doi.org link to every entry that has a DOI, and keep the `doi` field for other styles. Rating: 9/10. Switching to a DOI-aware style would also work, but it changes the look of the whole reference list for no extra benefit (5/10). hyperref is loaded at imports.tex:367 and xurl at imports.tex:379, so links will be clickable and will break across lines.

**`\nocite{*}` at thesis.tex:71.** Delete the line. Do it after adding `\citep{krizhevsky2012alexnet}` and `\citet{westbrook2016}`, and after deleting `sun_thesis_guide`, `sun_latex_template` and `kretzmann2022`.

**Page numbers that do not match the DOI.** These three entries pair CVF open-access pages with IEEE DOIs. Use the IEEE pages that Crossref gives: `huang2017densenet` 2261–2269 (was 4700–4708), `liu2022convnext` 11966–11976 (was 11976–11986), and `lin2017focal` 2999–3007 (was 2980–2988) ([Crossref DenseNet](https://doi.org/10.1109/CVPR.2017.243), [ConvNeXt](https://doi.org/10.1109/CVPR52688.2022.01167), [Focal](https://doi.org/10.1109/ICCV.2017.324)). Also add the missing issue numbers: `costa2013` number 9 plus DOI, and `lapuschkin2019` number 1.

**Brace protection.** IEEEtranN converts titles to sentence case, and thesis.bbl shows the damage. Use `{Green Drop}` for "Green drop national report 2022" (thesis.bbl:52). Use `{Inception}` for "Rethinking the inception architecture" (thesis.bbl:124). The other two damaged entries, "Stellenbosch university" (thesis.bbl:136) and the merged backtick in Kretzmann (thesis.bbl:142), disappear with their deletion. Protect proper nouns in every new entry too: `{Adam}`, `{PyTorch}`, `{TorchVision}`, `{VIA}`, `{ImageNet}`, `{ROC}`, `{South Africa}`, `{MobileNets}`, `{MobileNetV2}`. Existing protection of `{ImageNet}`, `{CNNs}`, `{Google Earth}`, `{OpenStreetMap}` and others is already correct. thesis.blg shows 0 warnings.

**Minor consistency.** Chapter5.tex:180 uses `\cite` where every other citation uses `\citep`. The output is identical under numeric natbib, but change it for tidiness. Move `geirhos2019`'s arXiv link from `note` to `url`. Make `hendrycks2016gelu` an `@misc` arXiv entry. Give `gildenblat2021pytorchcam` a `url` field and a version note. Online-only sources can keep `\url{...}` with "Accessed ..." in `note`, which prints correctly. Raw `%20` inside `\url` in a note was also tested and prints correctly.

**Google Earth imagery attribution.** This is the one item with a terms-of-use consequence, not just a style one. Google's Geo Guidelines require attribution on every use. It must name Google **and** the imagery providers, because "only including 'Google' or the Google logo is not proper attribution". It must stay "with the content" and must not be moved "to the end of your book" ([Google Geo Guidelines](https://about.google/brand-resource-center/products-and-services/geo-guidelines/)). Research and education use with attribution needs no request. Google says it "can't tell you if your use of this content would be fair use". No figure in the thesis currently carries any attribution: a grep of all chapters for "Google", "Maxar" and "Airbus" finds only body text. The affected figures are the background photos at Chapter2.tex:15–27, the facility overviews at Chapter4.tex:29–56, the class examples at Chapter4.tex:106 and 112, the relabel figures at Chapter4.tex:228 and 235, the augmentation figure at Chapter5.tex:107, and every Grad-CAM figure in Chapter 6. Check what basemap the map at Chapter4.tex:21 uses. The fix is to append "Imagery: Google Earth, © [year] [provider]" to each caption. Read the provider and year off the attribution in Google Earth Pro for each image, since they vary by date. Nobody could determine them remotely. For the small crops, one sentence at Chapter4.tex:4 plus a short caption note on each figure is defensible. Whether a caption pointer alone would satisfy Google is **uncertain**. A front-matter acknowledgement is a nice addition but does not replace per-figure attribution. Google Earth's own [Additional Terms of Service](https://support.google.com/earth/answer/21417?hl=en) exist but were not read.

**Template leftovers outside the bibliography.** Fix these before submission. thesis.tex:23 still says "PhD (Electronic Engineering)". The acronym list contains "MFT, my funky thesis". Nomenclature placeholders remain. The declaration is dated "10 May 2024". The abstract is empty. The acknowledgements still say "Supervisor X", "Funding Y" and thank Leanne for the template.

## 8. Change list

### A. Bibliography-only changes (safe, mechanical)

1. Delete `\nocite{*}` from thesis.tex:71. Do this after steps B9 and B10 add the two new citations.
2. Delete the entries `sun_thesis_guide`, `sun_latex_template` and `kretzmann2022` from mybib.bib.
3. Add `url = {https://doi.org/<doi>}` to every entry with a `doi` field. The appendix has the full entries.
4. Fix pages: `huang2017densenet` 2261–2269, `liu2022convnext` 11966–11976, `lin2017focal` 2999–3007.
5. `costa2013`: add `number = {9}`, the DOI and its url. `lapuschkin2019`: add `number = {1}`.
6. Add DOIs and urls to `lecun1998gradient`, `deng2009imagenet`, `he2016resnet`, `huang2017densenet`, `szegedy2016inception`, `hu2018squeeze`, `liu2022convnext`, `akiba2019optuna` and `selvaraju2017gradcam`. Add the PMLR volume and url to `tan2019efficientnet`.
7. `dws2022greendrop`: replace the IRIS landing-page URL with the direct 2022 PDF and the Wayback copy, and brace the title.
8. `szegedy2016inception`: brace `{Inception}`.
9. `krizhevsky2012alexnet`: add the NeurIPS url and remove the unverified pages.
10. `hendrycks2016gelu`: convert to `@misc` with an arXiv url. `geirhos2019`: move the link to `url`. `gildenblat2021pytorchcam`: add `url` and `note = {Version 1.5.7}`.
11. Replace `eoportal_quickbird` with `esa_quickbird`, or at least drop the unconfirmed "European Space Agency" from `howpublished`. Update the key at Chapter3.tex:25.
12. Change `\cite{akiba2019optuna}` to `\citep{akiba2019optuna}` at Chapter5.tex:180.
13. Add the new entries from the appendix: the must-add set, and any should-add entries you accept.
14. Add `russakovsky2015imagenet` to the citations at Chapter2.tex:84 and Chapter5.tex:77. Add `kingma2015adam` at Chapter5.tex:180, `paszke2019pytorch` and `torchvision2016` at Chapter5.tex:55, `dutta2019vgg` at Chapter4.tex:61, and `bergstra2011tpe` at Chapter5.tex:185. These only insert `\citep{...}` into existing sentences.
15. Fix the typo "woud" at Chapter3.tex:46 and the agreement "distributions adds" at Chapter3.tex:39.

### B. Text changes needing the author's decision

1. Chapter2.tex:89: change "vanishing gradient" to "degradation" and add the one-sentence correction (Section 3.1). **Do not skip this one.**
2. Chapter3.tex:23: "brighter" to "darker", and soften "focus on" (Section 3.2).
3. Chapter3.tex:9: rewrite with correct direction and page numbers. Choose the DPW version or the municipal version; the municipal one is recommended (Section 3.3).
4. Chapter3.tex:44: "up to 28%", F1-score, no second person (Section 3.4).
5. Chapter3.tex:25: QuickBird resolution wording, and optionally "in one case study in Wuhan" (Section 3.6).
6. Chapter4.tex:17: "designed to treat", cite Knight Piésold, and drop "ranks among the largest in South Africa" (Section 3.7).
7. Chapter3.tex:35: hedge the Penatti inference (Section 3.8).
8. Chapter3.tex:46, 14, 39, 30: "source tag"; "mainly of low index values" plus the attributed transfer-learning sentence; "studied on"; "a single class label regardless of condition" (Section 3.9).
9. Chapter3.tex:30 and 52: add the Westbrook sentence, and make "no study" read "no study ... automatically classifies" (Section 3.9).
10. Chapter3.tex:23: "a pretrained AlexNet \citep{krizhevsky2012alexnet}".
11. Chapter1.tex:6: add the Green Drop and Graham sentence. For the sampling-frequency claim, the "not widely implemented" claim and the sensor-cost claim at Chapter1.tex:4–6, either cite the source your supervisor used or attribute them as personal communication.
12. Chapter2.tex:8: attribute the visual cues to the annotation guidelines.
13. Add Google Earth provider attribution to every satellite-image caption (Section 7).
14. Should-add citations in Chapters 2 and 5 (Section 6). Decide which to accept. The metrics, BN/LN, MobileNets/MobileNetV2 and ViT ones are the most likely to draw a marker's comment.
15. Name scikit-learn where the metrics are computed, and cite it once its link is checked.

## Conclusion

The thesis has a trustworthy bibliography and a less trustworthy set of citing sentences, and those are different problems. The researchers found no fake sources. They did find four places where the cited paper argues against the thesis, or the thesis argues against itself: ResNet, the clarifier brightness, the Green Drop gap direction, and the Kattenborn worst case. Those are what cost marks, because a marker who opens the link finds them in one click. The IEEEtranN `doi` issue means no marker can currently make that click. So the bibliography-only fixes mainly make the text checkable, and that makes fixing the text urgent. Fix Chapter2.tex:89 first; it is one paragraph.

The less obvious lesson is about Westbrook. The essay you nearly deleted is the closest precedent to your own project. Citing it turns "no study was found" from a claim a marker can puncture into a defended one: people have judged plant state from Google Earth by eye, and this thesis automates that judgement. That framing is stronger than claiming to be first.

## Appendix: corrected and new BibTeX entries

Paste these over the existing entries with the same key, and add the new ones. Entries not listed here (`westbrook2016`) need no change. Delete `sun_thesis_guide`, `sun_latex_template` and `kretzmann2022`. Where a field is marked unverified in a comment, check it or leave it out.

### Existing entries, corrected

```bibtex
@article{lecun1998gradient,
	author  = {LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
	title   = {Gradient-Based Learning Applied to Document Recognition},
	journal = {Proceedings of the IEEE},
	volume  = {86},
	number  = {11},
	pages   = {2278--2324},
	year    = {1998},
	doi     = {10.1109/5.726791},
	url     = {https://doi.org/10.1109/5.726791},
}

@inproceedings{krizhevsky2012alexnet,
	author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
	title     = {{ImageNet} Classification with Deep Convolutional Neural Networks},
	booktitle = {Advances in Neural Information Processing Systems},
	volume    = {25},
	year      = {2012},
	url       = {https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html},
}
% pages 1097--1105 removed: not shown on the proceedings page, unverified

@inproceedings{deng2009imagenet,
	author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
	title     = {{ImageNet}: A Large-Scale Hierarchical Image Database},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {248--255},
	year      = {2009},
	doi       = {10.1109/CVPR.2009.5206848},
	url       = {https://doi.org/10.1109/CVPR.2009.5206848},
}

@inproceedings{he2016resnet,
	author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
	title     = {Deep Residual Learning for Image Recognition},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {770--778},
	year      = {2016},
	doi       = {10.1109/CVPR.2016.90},
	url       = {https://doi.org/10.1109/CVPR.2016.90},
}

@inproceedings{huang2017densenet,
	author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
	title     = {Densely Connected Convolutional Networks},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {2261--2269},
	year      = {2017},
	doi       = {10.1109/CVPR.2017.243},
	url       = {https://doi.org/10.1109/CVPR.2017.243},
}

@inproceedings{szegedy2016inception,
	author    = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
	title     = {Rethinking the {Inception} Architecture for Computer Vision},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {2818--2826},
	year      = {2016},
	doi       = {10.1109/CVPR.2016.308},
	url       = {https://doi.org/10.1109/CVPR.2016.308},
}

@inproceedings{hu2018squeeze,
	author    = {Hu, Jie and Shen, Li and Sun, Gang},
	title     = {Squeeze-and-Excitation Networks},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {7132--7141},
	year      = {2018},
	doi       = {10.1109/CVPR.2018.00745},
	url       = {https://doi.org/10.1109/CVPR.2018.00745},
}

@inproceedings{tan2019efficientnet,
	author    = {Tan, Mingxing and Le, Quoc V.},
	title     = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
	booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
	series    = {Proceedings of Machine Learning Research},
	volume    = {97},
	pages     = {6105--6114},
	publisher = {PMLR},
	year      = {2019},
	url       = {https://proceedings.mlr.press/v97/tan19a.html},
}

@inproceedings{liu2022convnext,
	author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
	title     = {A {ConvNet} for the 2020s},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {11966--11976},
	year      = {2022},
	doi       = {10.1109/CVPR52688.2022.01167},
	url       = {https://doi.org/10.1109/CVPR52688.2022.01167},
}

@inproceedings{akiba2019optuna,
	author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
	title     = {{Optuna}: A Next-Generation Hyperparameter Optimization Framework},
	booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
	pages     = {2623--2631},
	year      = {2019},
	doi       = {10.1145/3292500.3330701},
	url       = {https://doi.org/10.1145/3292500.3330701},
}

@inproceedings{selvaraju2017gradcam,
	author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
	title     = {{Grad-CAM}: Visual Explanations from Deep Networks via Gradient-Based Localization},
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
	pages     = {618--626},
	year      = {2017},
	doi       = {10.1109/ICCV.2017.74},
	url       = {https://doi.org/10.1109/ICCV.2017.74},
}

@misc{gildenblat2021pytorchcam,
	author       = {Gildenblat, Jacob and contributors},
	title        = {{PyTorch} library for {CAM} methods},
	year         = {2021},
	howpublished = {GitHub repository},
	note         = {Version 1.5.7},
	url          = {https://github.com/jacobgil/pytorch-grad-cam},
}

@misc{hendrycks2016gelu,
	author        = {Hendrycks, Dan and Gimpel, Kevin},
	title         = {Gaussian Error Linear Units ({GELUs})},
	year          = {2016},
	eprint        = {1606.08415},
	archivePrefix = {arXiv},
	howpublished  = {arXiv preprint arXiv:1606.08415},
	doi           = {10.48550/arXiv.1606.08415},
	url           = {https://arxiv.org/abs/1606.08415},
}

@techreport{dws2022greendrop,
	author      = {{Department of Water and Sanitation}},
	title       = {{National Green Drop Report 2022}},
	type        = {Report},
	institution = {Department of Water and Sanitation},
	address     = {Pretoria, South Africa},
	year        = {2022},
	note        = "\url{https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf}. Archived at \url{https://web.archive.org/web/20220710190541/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf}. Accessed 16 October 2026",
}

@article{costa2013,
	author  = {Costa, J. C. and Mesquita, D. P. and Amaral, A. L. and Alves, M. M. and Ferreira, E. C.},
	title   = {Quantitative image analysis for the characterization of microbial aggregates in biological wastewater treatment: a review},
	journal = {Environmental Science and Pollution Research},
	volume  = {20},
	number  = {9},
	pages   = {5887--5912},
	year    = {2013},
	doi     = {10.1007/s11356-013-1824-5},
	url     = {https://doi.org/10.1007/s11356-013-1824-5},
}

@article{satoh2021,
	author  = {Satoh, H. and Kashimoto, Y. and Takahashi, N. and Tsujimura, T.},
	title   = {Deep learning-based morphology classification of activated sludge flocs in wastewater treatment plants},
	journal = {Environmental Science: Water Research \& Technology},
	volume  = {7},
	number  = {2},
	pages   = {298--305},
	year    = {2021},
	doi     = {10.1039/D0EW00908C},
	url     = {https://doi.org/10.1039/D0EW00908C},
}

@article{borzooei2024,
	author  = {Borzooei, Sina and Scabini, Leonardo and Miranda, Gisele and Daneshgar, Saba and Deblieck, Lukas and Bruno, Odemir and De Langhe, Piet and De Baets, Bernard and Nopens, Ingmar and Torfs, Elena},
	title   = {Evaluation of activated sludge settling characteristics from microscopy images with deep convolutional neural networks and transfer learning},
	journal = {Journal of Water Process Engineering},
	volume  = {64},
	pages   = {105692},
	year    = {2024},
	doi     = {10.1016/j.jwpe.2024.105692},
	url     = {https://doi.org/10.1016/j.jwpe.2024.105692},
}

@article{carballomato2024,
	author  = {Carballo Mato, Joaqu{\'\i}n and Gonz{\'a}lez V{\'a}zquez, Sonia and Fern{\'a}ndez {\'A}guila, Jes{\'u}s and Delgado Rodr{\'\i}guez, {\'A}ngel and Lin, Xin and Garabato G{\'a}ndara, Luc{\'\i}a and Sobreira Seoane, Juan and Silva Castro, Jose},
	title   = {Foam segmentation in wastewater treatment plants},
	journal = {Water},
	volume  = {16},
	number  = {3},
	pages   = {390},
	year    = {2024},
	doi     = {10.3390/w16030390},
	url     = {https://doi.org/10.3390/w16030390},
}

@article{gholizadeh2016,
	author  = {Gholizadeh, Mohammad Haji and Melesse, Assefa M. and Reddi, Lakshmi},
	title   = {A comprehensive review on water quality parameters estimation using remote sensing techniques},
	journal = {Sensors},
	volume  = {16},
	number  = {8},
	pages   = {1298},
	year    = {2016},
	doi     = {10.3390/s16081298},
	url     = {https://doi.org/10.3390/s16081298},
}

@article{lopezbetancur2022,
	author  = {Lopez-Betancur, Daniela and Moreno, Ivan and Guerrero-Mendez, Carlos and Saucedo-Anaya, Tonatiuh and Gonz{\'a}lez, Efr{\'e}n and Bautista-Capetillo, Carlos and Gonz{\'a}lez-Trinidad, Juli{\'a}n},
	title   = {Convolutional neural network for measurement of suspended solids and turbidity},
	journal = {Applied Sciences},
	volume  = {12},
	number  = {12},
	pages   = {6079},
	year    = {2022},
	doi     = {10.3390/app12126079},
	url     = {https://doi.org/10.3390/app12126079},
}

@article{lesiv2018,
	author  = {Lesiv, Myroslava and See, Linda and {Laso Bayas}, Juan Carlos and Sturn, Tobias and Schepaschenko, Dmitry and Karner, Mathias and Moorthy, Inian and McCallum, Ian and Fritz, Steffen},
	title   = {Characterizing the spatial and temporal availability of very high resolution satellite imagery in {Google Earth} and {Microsoft Bing Maps} as a source of reference data},
	journal = {Land},
	volume  = {7},
	number  = {4},
	pages   = {118},
	year    = {2018},
	doi     = {10.3390/land7040118},
	url     = {https://doi.org/10.3390/land7040118},
}

% Replaces eoportal_quickbird (change \citep{eoportal_quickbird} to \citep{esa_quickbird} at Chapter3.tex:25)
@misc{esa_quickbird,
	author       = {{European Space Agency}},
	title        = {{QuickBird-2}},
	howpublished = {ESA Earth Online},
	note         = "\url{https://earth.esa.int/eogateway/missions/quickbird-2}. Accessed 16 October 2026",
}

% If you keep eoPortal instead, use this:
@misc{eoportal_quickbird,
	author       = {{eoPortal}},
	title        = {{QuickBird-2}},
	howpublished = {eoPortal satellite missions directory. Last updated 8 January 2025},
	year         = {2025},
	note         = "\url{https://www.eoportal.org/satellite-missions/quickbird-2}. Accessed 6 October 2026",
}

@article{hu2013,
	author  = {Hu, Qiong and Wu, Wenbin and Xia, Tian and Yu, Qiangyi and Yang, Peng and Li, Zhengguo and Song, Qian},
	title   = {Exploring the use of {Google Earth} imagery and object-based methods in land use/cover mapping},
	journal = {Remote Sensing},
	volume  = {5},
	number  = {11},
	pages   = {6026--6042},
	year    = {2013},
	doi     = {10.3390/rs5116026},
	url     = {https://doi.org/10.3390/rs5116026},
}

@article{zhou2018patternnet,
	author  = {Zhou, Weixun and Newsam, Shawn and Li, Congmin and Shao, Zhenfeng},
	title   = {{PatternNet}: A benchmark dataset for performance evaluation of remote sensing image retrieval},
	journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
	volume  = {145},
	pages   = {197--209},
	year    = {2018},
	doi     = {10.1016/j.isprsjprs.2018.01.004},
	url     = {https://doi.org/10.1016/j.isprsjprs.2018.01.004},
}

@article{li2022wwtp,
	author  = {Li, Hao and Zech, Johannes and Hong, Danfeng and Ghamisi, Pedram and Schultz, Michael and Zipf, Alexander},
	title   = {Leveraging {OpenStreetMap} and multimodal remote sensing data with joint deep learning for wastewater treatment plants detection},
	journal = {International Journal of Applied Earth Observation and Geoinformation},
	volume  = {110},
	pages   = {102804},
	year    = {2022},
	doi     = {10.1016/j.jag.2022.102804},
	url     = {https://doi.org/10.1016/j.jag.2022.102804},
}

@article{shuai2023,
	author  = {Shuai, Yue and Xie, Jun and Lu, Kaixuan and Chen, Zhengchao},
	title   = {Multi-attention network for sewage treatment plant detection},
	journal = {Sustainability},
	volume  = {15},
	number  = {7},
	pages   = {5880},
	year    = {2023},
	doi     = {10.3390/su15075880},
	url     = {https://doi.org/10.3390/su15075880},
}

@inproceedings{randhawa2023,
	author    = {Randhawa, Sukanya and Randhawa, Guntaj and Sivak, Olena and Zech, Johannes and Martin, Maria and Zipf, Alexander and Li, Yuze},
	title     = {Multiscale multifeature vision learning for scalable and efficient wastewater treatment plant detection using hi-res satellite imagery and {OSM}},
	booktitle = {Proceedings of the 1st ACM SIGSPATIAL International Workshop on Advances in Urban-AI (UrbanAI '23)},
	pages     = {10--21},
	publisher = {Association for Computing Machinery},
	address   = {New York, NY, USA},
	year      = {2023},
	doi       = {10.1145/3615900.3628772},
	url       = {https://doi.org/10.1145/3615900.3628772},
}

@article{handannader2019,
	author  = {Handan-Nader, Cassandra and Ho, Daniel E.},
	title   = {Deep learning to map concentrated animal feeding operations},
	journal = {Nature Sustainability},
	volume  = {2},
	number  = {4},
	pages   = {298--306},
	year    = {2019},
	doi     = {10.1038/s41893-019-0246-x},
	url     = {https://doi.org/10.1038/s41893-019-0246-x},
}

@article{nogueira2017,
	author  = {Nogueira, Keiller and Penatti, Ot{\'a}vio A. B. and dos Santos, Jefersson A.},
	title   = {Towards better exploiting convolutional neural networks for remote sensing scene classification},
	journal = {Pattern Recognition},
	volume  = {61},
	pages   = {539--556},
	year    = {2017},
	doi     = {10.1016/j.patcog.2016.07.001},
	url     = {https://doi.org/10.1016/j.patcog.2016.07.001},
}

@inproceedings{penatti2015,
	author    = {Penatti, Ot{\'a}vio A. B. and Nogueira, Keiller and dos Santos, Jefersson A.},
	title     = {Do deep features generalize from everyday objects to remote sensing and aerial scenes domains?},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
	pages     = {44--51},
	year      = {2015},
	doi       = {10.1109/CVPRW.2015.7301382},
	url       = {https://doi.org/10.1109/CVPRW.2015.7301382},
}

@inproceedings{geirhos2019,
	author    = {Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A. and Brendel, Wieland},
	title     = {{ImageNet}-trained {CNNs} are biased towards texture; increasing shape bias improves accuracy and robustness},
	booktitle = {International Conference on Learning Representations (ICLR)},
	year      = {2019},
	url       = {https://arxiv.org/abs/1811.12231},
}

@article{buda2018,
	author  = {Buda, Mateusz and Maki, Atsuto and Mazurowski, Maciej A.},
	title   = {A systematic study of the class imbalance problem in convolutional neural networks},
	journal = {Neural Networks},
	volume  = {106},
	pages   = {249--259},
	year    = {2018},
	doi     = {10.1016/j.neunet.2018.07.011},
	url     = {https://doi.org/10.1016/j.neunet.2018.07.011},
}

@inproceedings{lin2017focal,
	author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
	title     = {Focal Loss for Dense Object Detection},
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
	pages     = {2999--3007},
	year      = {2017},
	doi       = {10.1109/ICCV.2017.324},
	url       = {https://doi.org/10.1109/ICCV.2017.324},
}

@article{kattenborn2022,
	author  = {Kattenborn, Teja and Schiefer, Felix and Frey, Julian and Feilhauer, Hannes and Mahecha, Miguel D. and Dormann, Carsten F.},
	title   = {Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks},
	journal = {ISPRS Open Journal of Photogrammetry and Remote Sensing},
	volume  = {5},
	pages   = {100018},
	year    = {2022},
	doi     = {10.1016/j.ophoto.2022.100018},
	url     = {https://doi.org/10.1016/j.ophoto.2022.100018},
}

@article{geirhos2020,
	author  = {Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
	title   = {Shortcut learning in deep neural networks},
	journal = {Nature Machine Intelligence},
	volume  = {2},
	number  = {11},
	pages   = {665--673},
	year    = {2020},
	doi     = {10.1038/s42256-020-00257-z},
	url     = {https://doi.org/10.1038/s42256-020-00257-z},
}

@article{lapuschkin2019,
	author  = {Lapuschkin, Sebastian and W{\"a}ldchen, Stephan and Binder, Alexander and Montavon, Gr{\'e}goire and Samek, Wojciech and M{\"u}ller, Klaus-Robert},
	title   = {Unmasking {Clever Hans} predictors and assessing what machines really learn},
	journal = {Nature Communications},
	volume  = {10},
	number  = {1},
	pages   = {1096},
	year    = {2019},
	doi     = {10.1038/s41467-019-08987-4},
	url     = {https://doi.org/10.1038/s41467-019-08987-4},
}

@article{saito2015,
	author  = {Saito, Takaya and Rehmsmeier, Marc},
	title   = {The precision-recall plot is more informative than the {ROC} plot when evaluating binary classifiers on imbalanced datasets},
	journal = {PLoS ONE},
	volume  = {10},
	number  = {3},
	pages   = {e0118432},
	year    = {2015},
	doi     = {10.1371/journal.pone.0118432},
	url     = {https://doi.org/10.1371/journal.pone.0118432},
}
```

### New entries: must add

```bibtex
@article{russakovsky2015imagenet,
	author  = {Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and Berg, Alexander C. and Fei-Fei, Li},
	title   = {{ImageNet} Large Scale Visual Recognition Challenge},
	journal = {International Journal of Computer Vision},
	volume  = {115},
	number  = {3},
	pages   = {211--252},
	year    = {2015},
	doi     = {10.1007/s11263-015-0816-y},
	url     = {https://doi.org/10.1007/s11263-015-0816-y},
}

@inproceedings{kingma2015adam,
	author    = {Kingma, Diederik P. and Ba, Jimmy},
	title     = {{Adam}: A Method for Stochastic Optimization},
	booktitle = {International Conference on Learning Representations (ICLR)},
	year      = {2015},
	doi       = {10.48550/arXiv.1412.6980},
	url       = {https://doi.org/10.48550/arXiv.1412.6980},
}

@inproceedings{paszke2019pytorch,
	author    = {Paszke, Adam and others},
	title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year      = {2019},
	doi       = {10.48550/arXiv.1912.01703},
	url       = {https://doi.org/10.48550/arXiv.1912.01703},
}
% volume and pages not verified; left out

@misc{torchvision2016,
	author       = {{TorchVision maintainers and contributors}},
	title        = {{TorchVision}: {PyTorch}'s Computer Vision library},
	year         = {2016},
	howpublished = {GitHub repository},
	note         = {Version 0.28.0},
	url          = {https://github.com/pytorch/vision},
}

@inproceedings{dutta2019vgg,
	author    = {Dutta, Abhishek and Zisserman, Andrew},
	title     = {The {VIA} Annotation Software for Images, Audio and Video},
	booktitle = {Proceedings of the 27th ACM International Conference on Multimedia (MM '19)},
	publisher = {ACM},
	year      = {2019},
	doi       = {10.1145/3343031.3350535},
	url       = {https://doi.org/10.1145/3343031.3350535},
}
% pages not verified; the VIA site gives a ready-made BibTeX entry you can compare against

@inproceedings{bergstra2011tpe,
	author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
	title     = {Algorithms for Hyper-Parameter Optimization},
	booktitle = {Advances in Neural Information Processing Systems},
	volume    = {24},
	pages     = {2546--2554},
	year      = {2011},
	url       = {https://proceedings.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf},
}

@article{graham2025,
	author  = {Graham and Pattinson and Still},
	title   = {The state of wastewater management in {South Africa}: data gaps, missing wastewater, and {Green Drop} reporting},
	journal = {Water SA},
	volume  = {51},
	number  = {2},
	year    = {2025},
	doi     = {10.17159/wsa/2025.v51.i2.4152},
	url     = {https://doi.org/10.17159/wsa/2025.v51.i2.4152},
}
% Only surnames were verified. Add given names or initials and pages from the article page before compiling.

@misc{knightpiesold_northern,
	author       = {{Knight Pi{\'e}sold}},
	title        = {Northern Wastewater Treatment Works},
	howpublished = {Project page},
	note         = "\url{https://knightpiesold.com/en/projects/northern-wastewater-treatment-works/}. Accessed 16 October 2026",
}
```

### New entries: should add (paste only those you accept)

```bibtex
@article{sokolova2009,
	author  = {Sokolova, Marina and Lapalme, Guy},
	title   = {A systematic analysis of performance measures for classification tasks},
	journal = {Information Processing \& Management},
	volume  = {45},
	number  = {4},
	year    = {2009},
	doi     = {10.1016/j.ipm.2009.03.002},
	url     = {https://doi.org/10.1016/j.ipm.2009.03.002},
}
% pages not verified; add from the DOI page

@article{fawcett2006,
	author  = {Fawcett, Tom},
	title   = {An introduction to {ROC} analysis},
	journal = {Pattern Recognition Letters},
	volume  = {27},
	number  = {8},
	year    = {2006},
	doi     = {10.1016/j.patrec.2005.10.010},
	url     = {https://doi.org/10.1016/j.patrec.2005.10.010},
}
% pages not verified; add from the DOI page

@inproceedings{ioffe2015batchnorm,
	author    = {Ioffe, Sergey and Szegedy, Christian},
	title     = {Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift},
	booktitle = {Proceedings of the 32nd International Conference on Machine Learning (ICML)},
	year      = {2015},
	url       = {https://arxiv.org/abs/1502.03167},
}
% only the arXiv record was verified; the ICML venue is from general knowledge

@misc{ba2016layernorm,
	author       = {Ba, Jimmy Lei and Kiros, Jamie Ryan and Hinton, Geoffrey E.},
	title        = {Layer Normalization},
	year         = {2016},
	howpublished = {arXiv preprint arXiv:1607.06450},
	url          = {https://arxiv.org/abs/1607.06450},
}

@misc{howard2017mobilenets,
	author       = {Howard, Andrew G. and others},
	title        = {{MobileNets}: Efficient Convolutional Neural Networks for Mobile Vision Applications},
	year         = {2017},
	howpublished = {arXiv preprint arXiv:1704.04861},
	url          = {https://arxiv.org/abs/1704.04861},
}

@inproceedings{sandler2018mobilenetv2,
	author    = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
	title     = {{MobileNetV2}: Inverted Residuals and Linear Bottlenecks},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year      = {2018},
	doi       = {10.1109/CVPR.2018.00474},
	url       = {https://doi.org/10.1109/CVPR.2018.00474},
}
% pages not verified; add from Crossref

@inproceedings{dosovitskiy2021vit,
	author    = {Dosovitskiy, Alexey and others},
	title     = {An Image is Worth 16x16 Words: {Transformers} for Image Recognition at Scale},
	booktitle = {International Conference on Learning Representations (ICLR)},
	year      = {2021},
	url       = {https://arxiv.org/abs/2010.11929},
}

@inproceedings{micikevicius2018mixed,
	author    = {Micikevicius, Paulius and others},
	title     = {Mixed Precision Training},
	booktitle = {International Conference on Learning Representations (ICLR)},
	year      = {2018},
	url       = {https://arxiv.org/abs/1710.03740},
}

@misc{pytorch_reproducibility,
	author       = {{PyTorch Contributors}},
	title        = {Reproducibility},
	howpublished = {PyTorch documentation},
	note         = "\url{https://docs.pytorch.org/docs/stable/notes/randomness.html}. Accessed 16 October 2026",
}

@misc{pytorch_crossentropy,
	author       = {{PyTorch Contributors}},
	title        = {{CrossEntropyLoss}},
	howpublished = {PyTorch documentation},
	note         = "\url{https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html}. Accessed 16 October 2026",
}

@misc{torchvision_models,
	author       = {{TorchVision maintainers and contributors}},
	title        = {Models and pre-trained weights},
	howpublished = {TorchVision documentation},
	note         = "\url{https://docs.pytorch.org/vision/stable/models.html}. Accessed 16 October 2026",
}

@article{shorten2019augmentation,
	author  = {Shorten, Connor and Khoshgoftaar, Taghi M.},
	title   = {A survey on Image Data Augmentation for Deep Learning},
	journal = {Journal of Big Data},
	volume  = {6},
	pages   = {60},
	year    = {2019},
	doi     = {10.1186/s40537-019-0197-0},
	url     = {https://doi.org/10.1186/s40537-019-0197-0},
}

@incollection{prechelt1998early,
	author    = {Prechelt, Lutz},
	title     = {Early Stopping - But When?},
	booktitle = {Neural Networks: Tricks of the Trade},
	series    = {Lecture Notes in Computer Science},
	volume    = {1524},
	publisher = {Springer},
	year      = {1998},
	doi       = {10.1007/3-540-49430-8_3},
	url       = {https://doi.org/10.1007/3-540-49430-8_3},
}
% pages and editors not verified

@book{goodfellow2016deep,
	author    = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
	title     = {Deep Learning},
	publisher = {MIT Press},
	year      = {2016},
	url       = {https://www.deeplearningbook.org/},
}
% URL given by the researcher but not explicitly reported as fetched

@book{chen2020bwt,
	editor    = {Chen, Guanghao and van Loosdrecht, Mark C. M. and Ekama, George A. and Brdjanovic, Damir},
	title     = {Biological Wastewater Treatment: Principles, Modeling and Design},
	edition   = {2nd},
	publisher = {IWA Publishing},
	year      = {2020},
	doi       = {10.2166/9781789060362},
	url       = {https://doi.org/10.2166/9781789060362},
}
% DOI confirmed via Crossref; open access NOT confirmed

% scikit-learn (Pedregosa et al., JMLR 12, 2011): no researcher verified a link or metadata.
% Look it up on jmlr.org before adding.
```
