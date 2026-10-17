# ML-method bibliography audit (mybib.bib, 15 keys)

Scope: akiba2019optuna, deng2009imagenet, gildenblat2021pytorchcam, he2016resnet, hendrycks2016gelu, hu2018squeeze, huang2017densenet, lecun1998gradient, liu2022convnext, szegedy2016inception, tan2019efficientnet, selvaraju2017gradcam, lin2017focal, buda2018, krizhevsky2012alexnet.

Method: bib metadata checked against Crossref (api.crossref.org/works/<doi>), DataCite (arXiv DOI), PMLR and NeurIPS proceedings page metadata, and the GitHub README. Claims checked by downloading the arXiv full text (or the image-net.org PDF for Deng 2009) and searching it for the passages quoted below. CVF open-access and ScienceDirect returned HTTP 403 to the fetch tool, so CVF page numbers could not be read directly; IEEE page numbers come from Crossref.

## Q0. Cross-cutting issues (read first)

### Takeaway
Three problems affect several entries. (1) The style `IEEEtranN` never prints the `doi` field, so none of the existing `doi=` fields show in the PDF and the marker gets no link. Put the link in a `url` field. (2) Three entries use CVF open-access page numbers, which differ from the IEEE Xplore pages that belong with the DOI; `lin2017focal` already pairs CVF pages with an IEEE DOI. (3) `\nocite{*}` puts 5 uncited entries into the reference list.

### Cited Findings
- `Report/imports.tex:396-397` loads `\usepackage[square,numbers,compress,sort]{natbib}` and `\bibliographystyle{IEEEtranN}`. Running `grep -i doi` on `/usr/share/texmf-dist/bibtex/bst/ieeetran/IEEEtranN.bst` finds nothing, while the file does define `FUNCTION {format.url}` (line 1970). So `doi=` is ignored and `url=` is printed. (Checked locally, no web source.)
- Crossref IEEE pages compared with the pages in the bib: DenseNet 2261-2269 (bib 4700-4708); ConvNeXt 11966-11976 (bib 11976-11986); Focal loss 2999-3007 (bib 2980-2988). ResNet, SE, Inception-v3 and Grad-CAM agree. — [Crossref DenseNet](https://doi.org/10.1109/CVPR.2017.243), [Crossref ConvNeXt](https://doi.org/10.1109/CVPR52688.2022.01167), [Crossref Focal](https://doi.org/10.1109/ICCV.2017.324)
- `thesis.tex:71` has `\nocite{*}`. Entries with no `\cite` in any chapter: sun_thesis_guide, sun_latex_template, krizhevsky2012alexnet, kretzmann2022, westbrook2016. (Local grep.)

### Inferences
- The CVF pages are not wrong as such, since they are the open-access proceedings' own numbering. But once a doi.org link points to IEEE Xplore, the pages should be the IEEE ones. The entries below use IEEE pages throughout.
- Each corrected entry below keeps `doi` (harmless, and useful to other styles) and adds `url = {https://doi.org/...}` so that IEEEtranN prints a link.
- Removing `\nocite{*}` is better than deleting entries one by one, because a stale entry will otherwise appear in the reference list without being cited. A marker who sees uncited references may take them as padding.

### Gaps
- I did not check that `hyperref` or `url` is loaded so that the `\url{}` output can be clicked (it very probably is).

## Q1. Per-key verdicts

### Takeaway
All 15 sources exist and the authors and titles are correct. Two claims are not supported as written: the **ResNet sentence on vanishing gradients** (Ch2:89), which the paper itself argues against, and **Deng 2009 for "over a million images, a thousand classes"** (Ch2:84), which describes the 1000-class ILSVRC subset from Russakovsky et al. 2015, not the 2009 database. Page-number fixes are needed for densenet, convnext and focal. GELU has no peer-reviewed version, so the arXiv entry is the right one. AlexNet should be deleted.

### Cited Findings

#### akiba2019optuna — OK (needs url)
- Thesis: Ch5:180 "...were tuned with Optuna~\cite{akiba2019optuna}, a hyperparameter optimisation library."
- Crossref: "Optuna", Akiba, Sano, Yanase, Ohta, Koyama, Proc. 25th ACM SIGKDD, pp. 2623-2631, 2019. Matches the bib. — [https://doi.org/10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)
- Claim: trivially supported (the paper is the Optuna framework paper). Checked against metadata and title only.
```bibtex
@inproceedings{akiba2019optuna,
	author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
	title     = {{Optuna}: A Next-Generation Hyperparameter Optimization Framework},
	booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
	pages     = {2623--2631},
	year      = {2019},
	doi       = {10.1145/3292500.3330701},
	url       = {https://doi.org/10.1145/3292500.3330701},
}
```

#### deng2009imagenet — metadata OK, claim NOT supported by this source alone
- Thesis: Ch2:84 "...pretrained on ImageNet \citep{deng2009imagenet}, a benchmark dataset of over a million labelled natural images spanning a thousand object classes..." and Ch5:77 "...weights pretrained on ImageNet~\citep{deng2009imagenet}, specifically torchvision's IMAGENET1K_V1 weights."
- Crossref: CVPR 2009, pp. 248-255, all six authors match. — [https://doi.org/10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)
- Full text (image-net.org PDF) gives the 2009 database as "12 subtrees ... 5247 synsets and 3.2 million images", aiming at "50 million" images. It never mentions a 1000-class subset. — [Deng et al. 2009 PDF](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf)
- The 1000-class benchmark comes from Russakovsky et al.: "ILSVRC2012 with 1000 object classes and 1,431,167 annotated images". — [https://doi.org/10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y) (Crossref: IJCV 115(3):211-252, 2015; full text checked via [arXiv:1409.0575](https://arxiv.org/abs/1409.0575))
- Recommendation: keep Deng and add Russakovsky to the Ch2:84 sentence, i.e. `\citep{deng2009imagenet, russakovsky2015imagenet}`. Ch5:77 should also cite Russakovsky, since IMAGENET1K means ILSVRC-2012. Checked against full text.
```bibtex
@inproceedings{deng2009imagenet,
	author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
	title     = {{ImageNet}: A Large-Scale Hierarchical Image Database},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {248--255},
	year      = {2009},
	doi       = {10.1109/CVPR.2009.5206848},
	url       = {https://doi.org/10.1109/CVPR.2009.5206848},
}

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
```
(Author family names were checked against Crossref. Given names are as commonly published; I did not check them one by one.)

#### gildenblat2021pytorchcam — OK (use the README's own form)
- Thesis: Ch5:404 "Grad-CAM was applied to the output of the last feature block of each selected model, using the pytorch-grad-cam library~\citep{gildenblat2021pytorchcam}... Every heatmap is normalised separately..."
- The README asks for this citation: `@misc{jacobgilpytorchcam, title={PyTorch library for CAM methods}, author={Jacob Gildenblat and contributors}, year={2021}, publisher={GitHub}, howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}}}`. The bib matches it. — [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- The claim that the library was used is fine. The README does not state that the CAMs are normalised per image (see Gaps).
```bibtex
@misc{gildenblat2021pytorchcam,
	author       = {Gildenblat, Jacob and contributors},
	title        = {{PyTorch} library for {CAM} methods},
	year         = {2021},
	howpublished = {GitHub repository},
	url          = {https://github.com/jacobgil/pytorch-grad-cam},
}
```
(Optional: add the version used, e.g. `note = {Version x.y.z}`, from `pip show grad-cam`.)

#### he2016resnet — metadata OK, ONE CLAIM CONTRADICTED BY THE PAPER
- Thesis: Ch2:89 "Residual Networks (ResNet) \citep{he2016resnet} address a specific problem for CNNs: gradients ... can shrink so much ... (the vanishing gradient problem). Beyond a certain depth, plain networks ... begin to perform worse, even on the training set..." Also Ch2:84, Ch2:190 (bottleneck 256→64→256), Ch3:37, Ch5:57 (ResNet-50).
- Crossref: CVPR 2016, pp. 770-778, matches the bib. — [https://doi.org/10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
- The paper says the opposite of the first sentence: "the notorious problem of vanishing/exploding gradients ... has been largely addressed by normalized initialization ... and intermediate normalization layers". It then says "We argue that this optimization difficulty is unlikely to be caused by vanishing gradients ... We also verify that the backward propagated gradients exhibit healthy norms with BN." The problem ResNet addresses is named: "we address the degradation problem by introducing a deep residual learning framework". — [arXiv:1512.03385 full text](https://arxiv.org/abs/1512.03385)
- The second sentence (worse training error with depth) is supported: "plain nets ... exhibit higher training error when the depth increases".
- The 1×1 projection shortcut is supported ("projection shortcut ... done by 1×1 convolutions", option B). The 256→64→256 bottleneck is in the paper's Fig. 5.
- Fix: rewrite Ch2:89 so that ResNet addresses the **degradation problem**. Vanishing gradients can be mentioned as the older problem that normalisation had already largely solved, but should not be given as ResNet's motivation. Ch2:91 ("a gradient ... always has a direct, unimpeded path") is a common explanation, but it is not the paper's argument. Checked against full text.
```bibtex
@inproceedings{he2016resnet,
	author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
	title     = {Deep Residual Learning for Image Recognition},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {770--778},
	year      = {2016},
	doi       = {10.1109/CVPR.2016.90},
	url       = {https://doi.org/10.1109/CVPR.2016.90},
}
```

#### hendrycks2016gelu — OK (arXiv is correct; no peer-reviewed version found)
- Thesis: Ch2:195 "Its authors describe this as weighting inputs by their value rather than gating them by their sign \citep{hendrycks2016gelu}."
- The abstract says: "The GELU nonlinearity weights inputs by their value, rather than gates inputs by their sign as in ReLUs". The GELU = xΦ(x) definition matches. Supported, checked against full text. — [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- DataCite confirms DOI 10.48550/arXiv.1606.08415 (Hendrycks, Gimpel, 2016, resolving to arxiv.org/abs/1606.08415). — [https://doi.org/10.48550/arXiv.1606.08415](https://doi.org/10.48550/arXiv.1606.08415)
- The `@article` with `journal = {arXiv preprint ...}` works, but `@misc` is cleaner.
```bibtex
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
```

#### hu2018squeeze — OK (needs url)
- Thesis: Ch2:156 "...a squeeze-and-excitation block \citep{hu2018squeeze}, which globally average-pools each channel and passes the result through two small fully-connected layers to produce a per-channel gate..."
- Crossref: CVPR 2018, pp. 7132-7141, Hu, Shen, Sun. Matches. — [https://doi.org/10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745)
- Full text: "This is achieved by using global average pooling to generate channel-wise statistics" and "a bottleneck with two fully-connected (FC) layers around the non-linearity". Supported, checked against full text. — [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- Crossref lists the venue as "IEEE/CVF Conference" for 2018; the bib's "IEEE Conference" is acceptable either way.
```bibtex
@inproceedings{hu2018squeeze,
	author    = {Hu, Jie and Shen, Li and Sun, Gang},
	title     = {Squeeze-and-Excitation Networks},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {7132--7141},
	year      = {2018},
	doi       = {10.1109/CVPR.2018.00745},
	url       = {https://doi.org/10.1109/CVPR.2018.00745},
}
```

#### huang2017densenet — metadata error (pages), claims OK
- Thesis: Ch2:127-134 (dense block, BN-ReLU-conv H_l, growth rate k=32, blocks 6/12/24/16, transition 1×1 conv halves channels + 2×2 avg pool). Also Ch2:84 and Ch3:37.
- Crossref: CVPR 2017, **pp. 2261-2269**; the bib has the CVF pages 4700-4708. — [https://doi.org/10.1109/CVPR.2017.243](https://doi.org/10.1109/CVPR.2017.243)
- Full text: "The growth rate for all the networks is k = 32 ... each 'conv' layer ... corresponds the sequence BN-ReLU-Conv"; transition = "batch normalization layer and an 1×1 convolutional layer followed by a 2×2 average pooling layer"; "k0 + k × (ℓ − 1) input feature-maps"; compression "θ = 0.5" (DenseNet-BC). All supported, checked against full text. — [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- Minor point: DenseNet-121 is the BC variant, so each H_l is BN-ReLU-Conv1×1-BN-ReLU-Conv3×3 (two convolutions), not "a convolution". The Ch2:132 wording ("including ... a convolution") understates this, but it is not false.
```bibtex
@inproceedings{huang2017densenet,
	author    = {Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q.},
	title     = {Densely Connected Convolutional Networks},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {2261--2269},
	year      = {2017},
	doi       = {10.1109/CVPR.2017.243},
	url       = {https://doi.org/10.1109/CVPR.2017.243},
}
```

#### lecun1998gradient — OK (needs url)
- Thesis: Ch2:37 "A Convolutional Neural Network (CNN) \citep{lecun1998gradient} is a type of artificial neural network designed for data with a grid-like structure..."
- Crossref: Proc. IEEE 86(11):2278-2324, 1998, LeCun, Bottou, Bengio, Haffner. Matches. — [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)
- Claim: this is the standard citation for CNNs (LeNet-5, shared weights). I checked metadata only, not the full text. The weight-sharing argument in the sentence is what the paper is known for. Low risk.
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
```

#### liu2022convnext — metadata error (pages), claims OK
- Thesis: Ch2:188-195 (modernised ResNet borrowing from ViTs; 7×7 depthwise; inverted bottleneck ×4 matching the Transformer MLP; depthwise conv moved up; BN→LN; ReLU→GELU; ConvNeXt-T 96→384→96). Also Ch2:84 and Ch3:37.
- Crossref: CVPR 2022, **pp. 11966-11976**; the bib has the CVF pages 11976-11986. — [https://doi.org/10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167)
- Full text: "the hidden dimension of the MLP block is four times wider than the input dimension"; "Moving up depthwise conv layer"; "the complex/inefficient modules (MSA, large-kernel conv) will have fewer channels, while the efficient, dense 1×1 layers will do the heavy lifting"; "Substituting BN with LN"; "Replacing ReLU with GELU"; "ConvNeXt-T: C = (96, 192, 384, 768), B = (3, 3, 9, 3)". All supported, checked against full text. — [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Minor point: "making training less sensitive to batch size" (Ch2:190) is not ConvNeXt's own reason. The paper only says BN "has many intricacies that can have a detrimental effect". It is a true general property of LN, but it is not attributed and should not be read as ConvNeXt's claim.
```bibtex
@inproceedings{liu2022convnext,
	author    = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
	title     = {A {ConvNet} for the 2020s},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {11966--11976},
	year      = {2022},
	doi       = {10.1109/CVPR52688.2022.01167},
	url       = {https://doi.org/10.1109/CVPR52688.2022.01167},
}
```

#### szegedy2016inception — OK (needs url)
- Thesis: Ch5:57 "The ResNet-50~\citep{he2016resnet} and Inception-v3~\citep{szegedy2016inception} architectures were also included in this evaluation..."
- Crossref: CVPR 2016, pp. 2818-2826, five authors match. — [https://doi.org/10.1109/CVPR.2016.308](https://doi.org/10.1109/CVPR.2016.308)
- Claim: this paper introduces Inception-v3, so the citation is correct. Checked against metadata (I did not download the full text).
- Note: commit 732e95b "removed inception and resnet 50" kept this sentence. If Inception-v3 is dropped from the thesis entirely, delete this entry and the sentence together.
```bibtex
@inproceedings{szegedy2016inception,
	author    = {Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
	title     = {Rethinking the Inception Architecture for Computer Vision},
	booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages     = {2818--2826},
	year      = {2016},
	doi       = {10.1109/CVPR.2016.308},
	url       = {https://doi.org/10.1109/CVPR.2016.308},
}
```

#### tan2019efficientnet — OK (add PMLR volume/url; no DOI exists)
- Thesis: Ch2:156-160 (compound scaling of depth/width/resolution; B0 from NAS; MBConv with SE; α·β²·γ² ≈ 2). Also Ch2:84 and Ch3:37.
- PMLR page metadata: title matches; authors Mingxing Tan, Quoc Le; pages 6105-6114; ICML 2019; PMLR. — [https://proceedings.mlr.press/v97/tan19a.html](https://proceedings.mlr.press/v97/tan19a.html)
- Full text: "we use neural architecture search to design a new baseline network"; "Its main building block is mobile inverted bottleneck MBConv ..., to which we also add squeeze-and-excitation optimization"; "s.t. α · β² · γ² ≈ 2". Supported, checked against full text. — [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- EfficientNet uses SiLU/Swish ("We also use SiLU (Swish-1) activation"), which fits the Ch2:63 wording "ReLU, or a smooth variant of it".
```bibtex
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
```

#### selvaraju2017gradcam — OK (needs url)
- Thesis: Ch3:46 "...Grad-CAM highlights the image regions that are important for the output of a CNN \citep{selvaraju2017gradcam}..." and Ch2:276-300 (gradient of the pre-softmax score y^c with respect to A^k, averaged over i,j to α_k^c; ReLU of the weighted sum; last conv layer).
- Crossref: ICCV 2017, pp. 618-626, six authors match. — [https://doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
- Full text: "we first compute the gradient of the score for class c, y^c (before the softmax), with respect to feature map activations A^k ... These gradients flowing back are global-average-pooled". On layer choice: "we can expect the last convolutional layers to have the best compromise between high-level semantics and detailed spatial information". Supported, checked against full text. — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Grad-CAM works on any conv layer; the last one is the paper's recommended default. Ch2:278's "Grad-CAM uses the output of the last convolutional layer" is slightly too absolute but acceptable.
- Typo in Ch3:46: "woud" should be "would".
```bibtex
@inproceedings{selvaraju2017gradcam,
	author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
	title     = {{Grad-CAM}: Visual Explanations from Deep Networks via Gradient-Based Localization},
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
	pages     = {618--626},
	year      = {2017},
	doi       = {10.1109/ICCV.2017.74},
	url       = {https://doi.org/10.1109/ICCV.2017.74},
}
```

#### lin2017focal — metadata error (pages do not match the DOI), claim OK
- Thesis: Ch3:39 "\citet{lin2017focal} proposed focal loss, which down-weights easy examples so that training focuses on hard ones. Both methods were developed on data with distinct classes, such as handwritten digits and everyday objects."
- Crossref: ICCV 2017, **pp. 2999-3007**, while the bib has the CVF pages 2980-2988 together with the IEEE DOI. Crossref's title is "Focal Loss for Dense Object Detection"; the bib's lower case "loss", "dense", "object", "detection" will print as-is under IEEEtran (sentence case is acceptable, but it differs from every other entry). — [https://doi.org/10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)
- Abstract: "reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples". Supported, checked against full text. — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Context: focal loss targets "the extreme foreground-background class imbalance encountered during training of dense detectors" on COCO. "Everyday objects" is fair. The thesis does not say the loss was for detection rather than image classification, and a marker might expect that.
```bibtex
@inproceedings{lin2017focal,
	author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
	title     = {Focal Loss for Dense Object Detection},
	booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
	pages     = {2999--3007},
	year      = {2017},
	doi       = {10.1109/ICCV.2017.324},
	url       = {https://doi.org/10.1109/ICCV.2017.324},
}
```

#### buda2018 — OK (needs url), small wording issue
- Thesis: Ch3:39 "\citet{buda2018} showed that class imbalance harms CNN performance and that oversampling was the best remedy in almost all their tests." Ch3:48 "Overall accuracy hides poor results on rare classes, which is why \citet{buda2018} used a multi-class version of the area under the ... ROC curve."
- Crossref: Neural Networks 106:249-259, Oct 2018, Buda, Maki, Mazurowski. Matches. — [https://doi.org/10.1016/j.neunet.2018.07.011](https://doi.org/10.1016/j.neunet.2018.07.011)
- Abstract: "(i) the effect of class imbalance on classification performance is detrimental; (ii) the method of addressing class imbalance that emerged as dominant in almost all analyzed scenarios was oversampling". Also "Our main evaluation metric is ... (ROC AUC) adjusted to multi-class tasks since overall accuracy metric is associated with notable difficulties in the context of imbalanced data". Datasets: "MNIST, CIFAR-10 and ImageNet". Both sentences are supported, checked against full text (arXiv version). — [arXiv:1710.05381](https://arxiv.org/abs/1710.05381)
- Wording: "Both methods were developed on data..." is inaccurate for Buda, who did not develop oversampling but tested it. Suggest "Both were studied on data...".
```bibtex
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
```

#### krizhevsky2012alexnet — exists, metadata OK; not cited; recommend DELETE
- The NeurIPS proceedings page metadata matches: title, authors Krizhevsky, Sutskever, Hinton E., Advances in Neural Information Processing Systems vol. 25, 2012. — [https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- It reaches the reference list only through `\nocite{*}` (thesis.tex:71).
- Where it could go: Ch2:63 ("All of the architectures used in this project use ... ReLU") or Ch2:84 (ImageNet pretraining). Neither sentence makes a claim that AlexNet supports. To use it, a new sentence would be needed (e.g. that AlexNet's ImageNet win made ImageNet-pretrained CNNs the standard starting point). That sentence is padding and does not help the argument. Rating for adding it: 3/10. Delete the entry and remove `\nocite{*}`.
- If kept anyway:
```bibtex
@inproceedings{krizhevsky2012alexnet,
	author    = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},
	title     = {{ImageNet} Classification with Deep Convolutional Neural Networks},
	booktitle = {Advances in Neural Information Processing Systems},
	volume    = {25},
	year      = {2012},
	url       = {https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html},
}
```
(Pages 1097-1105 are the commonly quoted range, but the proceedings page metadata I read gives no pages, so I could not verify them. Leave pages out or keep them knowingly.)

### Inferences
- Must fix: the ResNet framing (Ch2:89) and the Deng-only ImageNet size claim (Ch2:84). A marker who knows the ResNet paper will spot the first, because the paper explicitly says vanishing gradients are not the cause.
- Should fix: pages for huang2017densenet, liu2022convnext and lin2017focal; add `url` to every entry; remove `\nocite{*}`.
- Optional: "Both were studied on" (Ch3:39); the "woud" typo (Ch3:46); the LN batch-size remark not being ConvNeXt's claim.
- Version issues: none need a change. GELU only exists as arXiv. All others cite the peer-reviewed conference or journal version. SE-Net and Grad-CAM also have later journal extensions, but the conference versions are the original and appropriate.

### Gaps
- I could not open CVF open-access pages (HTTP 403). The claim that 4700-4708 / 11976-11986 / 2980-2988 are CVF pagination comes from the known CVF numbering scheme, not from reading those pages. The IEEE pages are confirmed by Crossref.
- I did not confirm from the source code that pytorch-grad-cam normalises each CAM separately to [0,1] (Ch5:404); the README does not say so. Check `pytorch_grad_cam/utils/image.py` (`scale_cam_image`) in the installed version.
- LeCun 1998 and Szegedy 2016 were checked on metadata only (no full-text search). Their uses in the thesis are generic, so the risk is low.
- I did not check the journal extensions of SE-Net (TPAMI) and Grad-CAM (IJCV), so no DOIs are given for them.
