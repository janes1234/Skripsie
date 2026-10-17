# Remote sensing, scene classification and shortcut-learning references: audit

Scope: geirhos2019, geirhos2020, lapuschkin2019, kattenborn2022, nogueira2017, penatti2015, zhou2018patternnet, saito2015, lesiv2018, costa2013, hu2013.
Bib file: /home/janes/Repos/Skripsie/Report/mybib.bib. All citing sentences are in /home/janes/Repos/Skripsie/Report/chapters/Chapter3.tex.
Metadata was checked against the Crossref API (api.crossref.org/works/<doi>) for every DOI. Every DOI below was resolved with curl and returned a 302 redirect to the publisher.

## Summary table

| Key | Verdict | Link | Bib change needed | Claim checked against |
|---|---|---|---|---|
| geirhos2019 | OK | https://arxiv.org/abs/1811.12231 | Optional (url= instead of note) | Full text (arXiv) |
| geirhos2020 | OK | https://doi.org/10.1038/s42256-020-00257-z | No | Full text (arXiv 2004.07780) |
| lapuschkin2019 | OK (small wording issue) | https://doi.org/10.1038/s41467-019-08987-4 | Optional (number = {1}) | Full text (Nature Comms PDF) |
| kattenborn2022 | Partially supported (number overstated) | https://doi.org/10.1016/j.ophoto.2022.100018 | No | Full text (KIT repository PDF) |
| nogueira2017 | OK | https://doi.org/10.1016/j.patcog.2016.07.001 (OA: https://arxiv.org/abs/1602.01517) | No | Full text (arXiv) |
| penatti2015 | OK, but the inference drawn from it is a stretch | https://doi.org/10.1109/CVPRW.2015.7301382 | No | Full text (CVF open access PDF) |
| zhou2018patternnet | OK | https://doi.org/10.1016/j.isprsjprs.2018.01.004 (OA: https://arxiv.org/abs/1706.03424) | No | Full text (arXiv) |
| saito2015 | OK | https://doi.org/10.1371/journal.pone.0118432 | No | Title and metadata only (the claim restates the title) |
| lesiv2018 | OK | https://doi.org/10.3390/land7040118 | No | Abstract |
| costa2013 | Metadata error (missing issue and DOI) | https://doi.org/10.1007/s11356-013-1824-5 | Yes | Abstract-level only (search snippet) |
| hu2013 | OK | https://doi.org/10.3390/rs5116026 | No | Abstract |

None of the 11 looks fabricated. Every one exists, and authors, titles, venues, volumes and pages match Crossref.

## geirhos2019

### Takeaway
Verdict: OK. The entry is correct and the claim is supported.

### Cited Findings
- Thesis, Chapter3.tex:37: "\citet{geirhos2019} showed that ImageNet-trained CNNs rely on texture far more than people do, which they framed as a weakness for recognising objects."
- The abstract says "ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes, which is in stark contrast to human behavioural evidence". The paper reports 48,560 psychophysical trials across 97 observers, and shows that a shape-biased model (trained on Stylized-ImageNet) improves accuracy and robustness. This supports the "weakness" framing. — [arXiv 1811.12231](https://arxiv.org/abs/1811.12231)
- Published at ICLR 2019, as the arXiv abs page states. The entry's authors and title match. — [arXiv 1811.12231](https://arxiv.org/abs/1811.12231)
- Optional tidy-up: put the link in url= rather than note=. ICLR 2019 has no DOI or page numbers.
```bibtex
@inproceedings{geirhos2019,
	author    = {Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A. and Brendel, Wieland},
	title     = {{ImageNet}-trained {CNNs} are biased towards texture; increasing shape bias improves accuracy and robustness},
	booktitle = {International Conference on Learning Representations (ICLR)},
	year      = {2019},
	url       = {https://arxiv.org/abs/1811.12231},
}
```

### Inferences
- The thesis then argues that the texture bias may help with aeration zones. That is the author's own reasoning and is presented as such. Fine.

### Gaps
- I could not verify the OpenReview forum link because the site returned a browser-verification page, so I have not given one. The arXiv link works (HTTP 200).

## geirhos2020

### Takeaway
Verdict: OK. The metadata matches Crossref exactly (vol 2, issue 11, pp. 665-673, 2020) and the definition is paraphrased correctly.

### Cited Findings
- Thesis, Chapter3.tex:46: "\citet{geirhos2020} describe shortcut learning, in which a network finds a rule that works on the test set but fails under new conditions".
- The paper's own definition: "shortcuts are decision rules that perform well on i.i.d. test data but fail on o.o.d. tests". The thesis paraphrase is accurate. — [arXiv 2004.07780](https://arxiv.org/abs/2004.07780); [DOI](https://doi.org/10.1038/s42256-020-00257-z)
- The Nature MI version is paywalled. The arXiv version (2004.07780) is the open preprint and can go in the notes for markers.

### Inferences
- None needed.

### Gaps
- None.

## lapuschkin2019

### Takeaway
Verdict: OK. The metadata is correct. Crossref gives issue 1 and article number 1096, so number = {1} could be added. The horse example is real, but the paper calls the mark a "source tag", not a "copyright tag".

### Cited Findings
- Thesis, Chapter3.tex:46: "\citet{lapuschkin2019} give the well-known case of a classifier that recognised horses by a copyright tag common in horse photographs."
- The paper says the Fisher vector classifier on PASCAL VOC 2007 "focuses on a source tag present in about one-fifth of the horse figures. Removing the tag also removes the ability to classify the picture as a horse. Furthermore, inserting the tag on a car image changes the classification from car to horse." — [Nature Communications](https://doi.org/10.1038/s41467-019-08987-4)
- The word "copyright" does not appear in the PDF text I searched. The paper uses "source tag" throughout. Many secondary accounts describe the tag as a copyright watermark, so "copyright tag" is a common gloss, but "source tag" is the paper's term.
- The model was a Fisher vector (FV) classifier, not a CNN. The thesis says "a classifier", which is accurate.
- Optional bib change:
```bibtex
@article{lapuschkin2019,
	author  = {Lapuschkin, Sebastian and W{\"a}ldchen, Stephan and Binder, Alexander and Montavon, Gr{\'e}goire and Samek, Wojciech and M{\"u}ller, Klaus-Robert},
	title   = {Unmasking {Clever Hans} predictors and assessing what machines really learn},
	journal = {Nature Communications},
	volume  = {10},
	number  = {1},
	pages   = {1096},
	year    = {2019},
	doi     = {10.1038/s41467-019-08987-4},
}
```

### Inferences
- Suggested minimal rewording: "by a source tag (a watermark) present in about a fifth of the horse images".

### Gaps
- None.

## kattenborn2022

### Takeaway
Verdict: partially supported. The paper is real and the metadata is correct. The thesis turns "up to 28%" into a flat "28%", calls an F1-score "accuracy", and treats block cross-validation as the "actual" accuracy, which the paper only assumes.

### Cited Findings
- Thesis, Chapter3.tex:44: "When analysing drone imagery, choosing random test images made the model appear to be 28\% more accurate than it actually was. This happens because images taken from the same area and at the same time are too similar ... the model must be tested on entirely separate geographic locations, a rule that becomes even more important when you only have a small amount of data."
- Abstract: "Assuming that a block cross-validation provides a realistic model performance, a validation with randomly sampled holdouts overestimated the model performance by up to 28%. Smaller training sample size increased this optimism. Spatial autocorrelation among observations was significantly higher within than between different remote sensing acquisitions." — [ISPRS Open J. (OA)](https://doi.org/10.1016/j.ophoto.2022.100018)
- Results section: the optimism "amounted to 14.6%, 17.7% and 27.9% when using 40, 25, and 10 UAV-orthoimages for model training, respectively". The metric is the F1-score of a tree-species segmentation task. — [KIT repository full text](https://publikationen.bibliothek.kit.edu/1000156879)
- Discussion: "model evaluations based on small sample sizes are even more likely to report optimistic performance, if spatial independence is violated". This supports the small-data clause. — [KIT repository full text](https://publikationen.bibliothek.kit.edu/1000156879)
- Supported: drone (UAV) imagery; within-acquisition similarity; the small-data effect; the recommendation of spatial (block) cross-validation and independent acquisitions.

### Inferences
- 28% is the worst case, not the typical case. It applies to the smallest training set (10 orthoimages). With 40 orthoimages the figure was 14.6%.
- "More accurate" is loose, because the metric is F1 for segmentation. "Than it actually was" overstates what the paper claims: the paper treats block CV as realistic by assumption.
- It is unclear from the extracted text whether the 28% is a relative change or a difference in F1 points. The paper writes "%" for a difference in F1-scores, so it looks like percentage points, but I am not certain.
- Suggested minimal rewording: "choosing random test images made the model's F1-score appear up to 28% higher than with spatially separate test data". "Up to" could instead be "between 15% and 28%, depending on the amount of training data".
- Style note for the author: "how you select" and "when you only have" use the second person, which does not fit the dry style used elsewhere in the thesis.

### Gaps
- None material.

## nogueira2017

### Takeaway
Verdict: OK. The metadata matches Crossref (Pattern Recognition 61, 539-556, 2017) and the claim matches the paper.

### Cited Findings
- Thesis, Chapter3.tex:35: "\citet{nogueira2017} compared full training, fine-tuning and fixed feature extraction for six CNNs on three remote sensing datasets and found that fine-tuning tended to work best".
- The paper runs "experiments with six popular ConvNets using three remote sensing datasets" (UCMerced, RS19, Brazilian Coffee Scenes). It states: "Our results point that fine-tuning tends to be the best performing strategy". It also says full training "was not a good strategy" for the smaller datasets. — [arXiv 1602.01517](https://arxiv.org/abs/1602.01517); [DOI](https://doi.org/10.1016/j.patcog.2016.07.001)
- The Elsevier version is paywalled. The arXiv preprint is the open alternative.

### Inferences
- "A result that \citet{borzooei2024} confirm in a very different domain" is outside my scope and was not checked here.

### Gaps
- None.

## penatti2015

### Takeaway
Verdict: the factual sentence is OK. The inference drawn from it is a mild stretch: the coffee images are green/red/near-infrared composites, not RGB, and the authors give texture as well as colour as reasons.

### Cited Findings
- Thesis, Chapter3.tex:35: "ImageNet features gave the best results on aerial scenes but were beaten by a simple colour descriptor on multispectral scenes of coffee crops. Features learnt from everyday objects may therefore miss a cue when classes differ mainly in colour, as the clarifier classes do."
- Abstract: "ConvNets obtained the best results for aerial images, while for remote sensing, they performed well but were outperformed by low-level color descriptors, such as BIC." — [CVF open access PDF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Penatti_Do_Deep_Features_2015_CVPR_paper.pdf); [DOI](https://doi.org/10.1109/CVPRW.2015.7301382)
- Numbers: on UCMerced, Caffe scored 93.42% ± 1.00 and OverFeat about 90-91%. On Coffee, BIC scored 87.03% ± 1.17 and Caffe 84.82% ± 0.97, a gap of about 2.2 points. — [CVF PDF](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/papers/Penatti_Do_Deep_Features_2015_CVPR_paper.pdf)
- Dataset: SPOT imagery from 2005, using "only the green, red, and near-infrared bands". — same source
- The authors' explanation: the coffee dataset "is composed of finer and more homogeneous textures where the patterns are much more overlapping visually and more different than everyday objects. The color/spectral properties are also important in this dataset". They also write: "As for the Coffee dataset the differences between classes may be not only in texture but also in color properties, BIC could encode well such differences." — same source
- Metadata matches Crossref (CVPRW 2015, pp. 44-51).

### Inferences
- "Beaten by a simple colour descriptor" is accurate. Calling BIC "simple" is fair: it is a border/interior colour histogram.
- The step to "classes differ mainly in colour, as the clarifier classes do" goes further than the source. The "colour" in the coffee data includes near-infrared, and the authors blame texture homogeneity as much as colour. The gap is also small (2.2 points). A hedge such as "partly because colour (here including near-infrared) mattered" would be more accurate. Nogueira et al. 2017 later fine-tuned CNNs on the same coffee dataset and the CNNs did well, which weakens the caution further. That last point is my inference from the nogueira2017 paper above, not a quoted result.

### Gaps
- IEEE Xplore was not opened (paywalled). The CVF PDF is the open copy and is the better link for markers.

## zhou2018patternnet

### Takeaway
Verdict: OK. It has 38 classes and "wastewater treatment plant" is one of them. The metadata matches Crossref (ISPRS J. 145, 197-209, Nov 2018).

### Cited Findings
- Thesis, Chapter3.tex:30: "Wastewater treatment plant is one of 38 scene classes in the PatternNet benchmark, with one label per plant whatever its condition".
- The paper lists 38 classes ending "... transformer station and wastewater treatment plant", with 800 images per class. It also notes that some storage-tank images in other datasets would be better labelled as wastewater treatment plants, and cites the wastewater class as an example of high intra-class diversity. — [arXiv 1706.03424](https://arxiv.org/abs/1706.03424); [DOI](https://doi.org/10.1016/j.isprsjprs.2018.01.004)

### Inferences
- "One label per plant whatever its condition" is the thesis author's reading. It is true by construction, since PatternNet is a scene-level class with no condition labels, but the paper does not say it in those words. Acceptable.
- Minor point: the dataset contains 800 image chips per class, not one per plant. "One label per plant" is slightly loose. "A single class label regardless of condition" would be more precise.

### Gaps
- None.

## saito2015

### Takeaway
Verdict: OK. The metadata matches Crossref (PLoS ONE 10(3): e0118432, 2015), and the claim restates the paper's title.

### Cited Findings
- Thesis, Chapter3.tex:48: "\citet{saito2015} argue that precision-recall curves are more informative than ROC curves when classes are imbalanced".
- Title: "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets". — [PLoS ONE (OA)](https://doi.org/10.1371/journal.pone.0118432)

### Inferences
- The paper is about binary classifiers. The thesis uses it in a multi-class or one-vs-rest setting, which is a standard and acceptable extension.

### Gaps
- I did not re-read the body text. The claim needs only the title.

## lesiv2018

### Takeaway
Verdict: OK. The metadata matches Crossref (Land 7(4): 118, 2018), and the abstract supports all three claims almost word for word.

### Cited Findings
- Thesis, Chapter3.tex:25: "In remote sensing it is increasingly used to build reference data for land cover mapping through visual interpretation ... \citet{lesiv2018} mapped where and when such imagery exists worldwide and found that availability is biased towards regions such as the USA, Europe and India, with clear breaks at political borders."
- Abstract: "VHR satellite imagery from Google Earth and Microsoft Bing Maps is increasingly being used in a variety of applications ... In the field of remote sensing, one use of this imagery is to create reference data sets through visual interpretation, e.g., to complement existing training data or to aid in the validation of land-cover products." Also: "The results show an uneven availability globally, with biases in certain areas such as the USA, Europe and India, and with clear discontinuities at political borders." — [Land (OA)](https://doi.org/10.3390/land7040118)

### Inferences
- The "increasingly used" in the paper applies to applications in general. Reference-data creation is named as one use in remote sensing. The thesis compresses these two points slightly, which is acceptable.
- The paper covers Bing Maps as well as Google Earth. The thesis mentions only Google Earth, which is fine for its purpose.

### Gaps
- MDPI blocked automated full-text access (HTTP 403). The abstract, read via OpenAlex, was sufficient.

## costa2013

### Takeaway
Verdict: metadata error. The paper exists and the authors, title, volume and pages are correct, but the entry lacks issue = {9} and the DOI. The claim is broad and fits a review of this scope.

### Cited Findings
- Thesis, Chapter3.tex:14: "Quantitative image analysis of microscope images of flocs and filaments has grown into a field of its own \citep{costa2013}".
- Crossref: DOI 10.1007/s11356-013-1824-5, Environmental Science and Pollution Research 20(9), 5887-5912, 2013. Authors Costa, Mesquita, Amaral, Alves, Ferreira. — [Crossref/DOI](https://doi.org/10.1007/s11356-013-1824-5)
- The review covers image analysis for monitoring microbial entities, as single cells and as aggregates, in biological wastewater treatment. This includes descriptors that predict filamentous bulking in activated sludge and that classify protozoa and metazoa. — [RCAAP record](https://www.rcaap.pt/vufind/PublicationRecord/rum_bb12ac3c515bbdc72e4fc2bf0074760f) (search snippet only)
- Corrected entry:
```bibtex
@article{costa2013,
	author  = {Costa, J. C. and Mesquita, D. P. and Amaral, A. L. and Alves, M. M. and Ferreira, E. C.},
	title   = {Quantitative image analysis for the characterization of microbial aggregates in biological wastewater treatment: a review},
	journal = {Environmental Science and Pollution Research},
	volume  = {20},
	number  = {9},
	pages   = {5887--5912},
	year    = {2013},
	doi     = {10.1007/s11356-013-1824-5},
}
```

### Inferences
- A 25-page review dedicated to the topic supports "has grown into a field of its own". It also covers anaerobic granules and biofilms, not only flocs and filaments, but that does not undermine the claim.

### Gaps
- The Springer page redirected to a login, and the University of Minho handle (hdl.handle.net/1822/25500, listed by OpenAlex as an open-access copy) returned HTTP 500 when tested. I therefore could not read the abstract or full text directly. Claim support rests on the title, the venue and a search snippet.

## hu2013

### Takeaway
Verdict: OK. The metadata matches Crossref (Remote Sensing 5(11), 6026-6042, 2013), and the abstract supports the claim.

### Cited Findings
- Thesis, Chapter3.tex:25: "\citet{hu2013} compared Google Earth imagery for land cover mapping with imagery from QuickBird ... Google Earth was almost as accurate. It did well on classes that can be recognised by shape and context, but poorly on classes that can only be separated by colour, because it carries little spectral information."
- Abstract: "GE has an overall classification accuracy of 78.07%, which is slightly lower than that of QB. No significant difference was found between these two classification results by the adoption of Z-test". Also: "It possesses some advantages for mapping those types with good spatial characteristics in terms of geometric, shape and context ... However, GE has some limitations for those types classified by using only spectral characteristics largely due to its poor spectral characteristics." — [Remote Sensing (OA)](https://doi.org/10.3390/rs5116026)
- Context: a single case study in Wuhan City, object-based classification, 570 validation points. — same source

### Inferences
- "Only by colour" is a fair plain-language version of "only spectral characteristics". Strictly, QuickBird's advantage includes a near-infrared band, not just colour. The thesis's own QuickBird sentence ("four spectral bands") covers this.
- The thesis could add "in one case study in Wuhan" to avoid over-generalising. This is optional.

### Gaps
- MDPI blocked full-text access, so I did not get QuickBird's exact overall accuracy. The abstract says only "slightly lower".

## Other observations (outside the 11 keys, noticed in passing)
- Chapter3.tex:46 has the typo "woud" (should be "would").
