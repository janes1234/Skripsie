# Wastewater / water-quality / satellite-sensor bibliography audit

Scope: borzooei2024, carballomato2024, dws2022greendrop, eoportal_quickbird, gholizadeh2016, handannader2019, li2022wwtp, lopezbetancur2022, randhawa2023, satoh2021, shuai2023, plus the uncited kretzmann2022 and westbrook2016. All citations sit in `Report/chapters/Chapter3.tex`. Line numbers are paragraph lines in that file. Audit date: 16 October 2026. No Report files were edited.

Summary table

| key | verdict | verified link | checked against |
|---|---|---|---|
| borzooei2024 | OK (one small overstatement) | https://doi.org/10.1016/j.jwpe.2024.105692 | full text (arXiv v3 preprint, 2402.09367) |
| carballomato2024 | OK | https://doi.org/10.3390/w16030390 | full text (MDPI PDF) |
| dws2022greendrop | metadata error (URL) + claim partly misleading | https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf | full text (558-page PDF) |
| eoportal_quickbird | claim wrong in one detail; metadata minor | https://www.eoportal.org/satellite-missions/quickbird-2 | full page |
| gholizadeh2016 | partially supported; a nearby sentence contradicts the source | https://doi.org/10.3390/s16081298 | full text (PMC5017463) |
| handannader2019 | OK | https://doi.org/10.1038/s41893-019-0246-x | abstract only (paywalled) |
| li2022wwtp | OK | https://doi.org/10.1016/j.jag.2022.102804 | full text (PMC9626640) |
| lopezbetancur2022 | OK (one nuance worth adding) | https://doi.org/10.3390/app12126079 | full text (MDPI PDF) |
| randhawa2023 | OK | https://doi.org/10.1145/3615900.3628772 | abstract only (gold OA, I did not fetch the PDF) |
| satoh2021 | OK | https://doi.org/10.1039/D0EW00908C | abstract only (green OA copy exists at Hokkaido, could not download it) |
| shuai2023 | OK | https://doi.org/10.3390/su15075880 | full text (MDPI PDF) |
| kretzmann2022 | real, not cited; delete or cite secondarily | https://www.dailymaverick.co.za/article/2022-06-21-a-tsunami-of-human-waste-half-of-sas-sewage-treatment-works-are-failing-says-study/ | full article |
| westbrook2016 | real, not cited; cite it (closest precedent) | https://ascecubadatabase.org/asce_proceedings/havanas-wastewater-treatment-plants-changes-over-time-and-estimate-of-replacement-cost/ | full text (HTML) |

## Cross-cutting issue: no DOI link is printed in the reference list

### Takeaway
The thesis uses `\bibliographystyle{IEEEtranN}` (imports.tex:397). IEEEtranN ignores the `doi` field, so none of the 20 DOIs in mybib.bib appear in the PDF, and a marker gets no clickable link for any journal article. A `url` field is printed ("[Online]. Available: ...").

### Cited Findings
- thesis.bbl shows borzooei2024, satoh2021, carballomato2024, li2022wwtp etc. rendered with no DOI or URL, while `note = "\url{...}"` entries do show the URL. Checked in /home/janes/Repos/Skripsie/Report/thesis.bbl.
- A scratchpad test compile with the thesis's own package set (hyperref, url, xurl, natbib `[square,numbers,compress,sort]`, IEEEtranN) printed `url = {https://doi.org/10.3390/su15075880}` as "[Online]. Available: https://doi.org/10.3390/su15075880". A `note = "\url{...%20...}"` with a raw `%20` also compiled and printed correctly.
- The `\url` in a note field printed for the DWS entry; IEEEtranN also lower-cased the title to "Green drop national report 2022" (thesis.bbl line 52), so title words need braces.

### Inferences
- Simplest fix: add `url = {https://doi.org/<doi>}` next to each `doi` field. The corrected entries below do this.

### Gaps
- None.

## borzooei2024

### Takeaway
Real paper with correct metadata. All the specific claims check out against the text. One wording point: the paper says only that the underestimation "may be linked to" training data dominated by low SVI values. "A very small number of such samples" is a little stronger than what they wrote.

### Cited Findings
- Crossref: Borzooei, Scabini, Miranda, Daneshgar, Deblieck, Bruno, De Langhe, De Baets, Nopens, Torfs; J. Water Process Eng. 64, 105692, July 2024. Matches the bib entry. — [Crossref/DOI](https://doi.org/10.1016/j.jwpe.2024.105692)
- Thresholding claim (Ch3:14): "These semi-automatic techniques can be highly influenced by the subjectivity and bias of practitioners' decisions regarding the selection of algorithms and the pre-defined threshold values." — [arXiv 2402.09367v3](https://arxiv.org/abs/2402.09367v3)
- Data (Ch3:14): "collected over two years, with weekly sampling at a full-scale industrial WWTP in Belgium", from 1 Nov 2020 to 31 Jul 2022 (21 months, so "almost two years" is accurate). — [arXiv](https://arxiv.org/abs/2402.09367v3)
- Transfer learning (Ch3:14, Ch3:35): each CNN was "built from scratch or fine-tuned with pretraining on ImageNet-1k"; TL "starts from a lower loss for all CNN models, leading to a better final performance"; "TL is necessary for achieving better SVI fitting". — [arXiv](https://arxiv.org/abs/2402.09367v3)
- Overfitting and model size (Ch3:14, Ch3:37): "bigger CNN variants like ResNet152 and ConvNeXt-S were found to be more challenging to train due to the occurrence of overfitting ... achieving lower generalization performance than their smaller" variants; ConvNeXt-nano best, R² = 0.75 ± 0.05, MAE 37.51. — [arXiv](https://arxiv.org/abs/2402.09367v3)
- High SVI (Ch3:14): the model "struggles with samples with SVI values above a certain threshold (400 ml/g) ... actual SVI values often exceed the model's predictions. This issue may be linked to a bias in the training data, which mainly includes instances of lower SVI values". — [arXiv](https://arxiv.org/abs/2402.09367v3)
- Single plant (Ch3:18): all data come from one industrial plant, as the thesis says.

### Inferences
- Optional minimal rewording of Ch3:14: "...which the authors link to a training set made up mainly of low index values."
- I checked the arXiv v3 preprint, not the Elsevier version of record. v1 had a different title ("Prediction of..."), but v3 has the journal title, so the content should match.

### Corrected BibTeX (adds a printable link only)
```bibtex
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
```

### Gaps
- I did not compare against the Elsevier version of record.

## carballomato2024

### Takeaway
Real, correct metadata, and the claims (PTZ camera, texture segmentation, 86% Dice, real-time foam percentage, notices to operators) are all supported.

### Cited Findings
- Crossref matches every field (Water 16(3), 390, 24 Jan 2024, eight authors as in the bib). — [DOI](https://doi.org/10.3390/w16030390)
- Abstract: "monitor in real time the presence of foams in WWTP tanks by using texture segmentation models trained with centralized and federated approaches. These models are designed to segment the foam and quantify the percentage of foam coverage"; "acquiring images using a PTZ camera ... implementing a real-time communication method for event notifications to plant operators"; "achieving an 86% Dice score in foam segmentation". — [MDPI PDF](https://mdpi-res.com/d_attachment/water/water-16-00390/article_deploy/water-16-00390.pdf)
- Images were taken "in 2 tanks of the WWTP" with a Hanwha PTZ camera, so "at a working plant" is correct. The model is One Shot Texture Segmentation (OSTS), which supports "treated tank foam as a texture problem" (Ch3:37). — same PDF

### Inferences
- Ch3:16 "reported the percentage ... to the operators in real time" is a fair compression. The paper describes event notifications to operators plus a computed foam percentage.

### Corrected BibTeX
```bibtex
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
```

### Gaps
- None.

## dws2022greendrop

### Takeaway
The report is real, and the numbers 115, 102 and 59% are all in it. Three problems, though:
1. The URL in the bib is the IRIS landing page, which now shows the Green Drop **2025** report. A marker following it will not find the 2022 report.
2. The thesis says "the 2022 Green Drop report for the Department of Public Works". The cited title is the National report. DWS also published a separate DPW report. The national report contains the DPW chapter (ch. 14), so citing the national report works, but the sentence should give the page.
3. Direction problem. For DPW, the 59% gap is a case where the physical site score was *higher* than the desktop (paperwork) score: Waterval Prison scored 0% on paper and 59% on site. The thesis presents paperwork as hiding poor performance, which is the opposite of what these DPW numbers show. DPW systems are also prisons, border posts and similar state facilities, not municipal works like the Cape Town plants. The report has better municipal evidence.

### Cited Findings
- Thesis Ch3:9: "The 2022 Green Drop report for the Department of Public Works shows a clear gap between official records and actual plant conditions. Out of 115 systems, 102 are in a critical state. The report notes a disconnect between administrative data and physical fieldwork. Physical inspections found differences of up to 59% between paper-based scores and the real state of the facilities. ... scores based on submitted paperwork can hide true performance."
- 115 / 102: "Unfortunately one hundred and two (102 of 115) systems were identified in critical state in the DPW, compared to 104 of 121 systems in 2013." National report p. 451 (DPW synopsis); also exec summary p. xi: "A total of 102 (89%) out of the 115 DPW systems were identified in critical state". Critical state means a Green Drop score below 31%. — [DWS national report PDF](https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf)
- Disconnect wording: "A deviation of >10% between the GD and TSA score indicate a misalignment between the administrative aspects and the work on the ground." (p. 469 and the same wording in every provincial chapter). — same PDF
- 59%: "All the remaining DPW Regions had large deviations ranging from 21% to 59%" (p. 469). Table 16 / Fig. 16 (pp. 468–469): Waterval Prison GD 0%, TSA 59%; Devon GD 22%, TSA 66%; St Albans GD 42%, TSA 81%. In almost every DPW case the TSA (site) score is above the GD (desktop) score. — same PDF
- What the scores are: the TSA is "a physical inspection ... done at 1 to 2 sites to confirm the findings of the desktop audit", reflecting "the physical condition of the sewer collector network, pumping stations, treatment plant and point of discharge" (exec summary). — same PDF
- Municipal figures (more relevant to Cape Town): "A total of 334 (39%) of municipal wastewater systems were identified to be in a critical state in 2021, compared to 248 (29%) in 2013" (p. xi); the audit covered 995 systems (850 municipal, 115 DPW, 30 private). — same PDF
- Municipal GD/TSA mismatch: 155 TSAs nationally, deviation 0% to 57%; "KwaZulu Natal had the highest %deviation of 56% ... indicating a severe disconnect between wastewater administration and field conditions" (p. 32). — same PDF
- Cape Town context: "City of Cape Town, Drakenstein, Stellenbosch ... had TSA scores >80%, which also include a close match to their respective GD scores" (p. 31). City of Cape Town is listed among Green Drop certified institutions (p. x). — same PDF
- The national report says itself that "The DPW Green Drop Report 2022 is a sub-set of the national report". The separate DPW report is also live: https://ws.dws.gov.za/IRIS/releases/GD22%20Report_DPW_Rev03_05May22_MN%20web.pdf (HTTP 200 on 16 Oct 2026; Wayback copy https://web.archive.org/web/20220710190925/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_DPW_Rev03_05May22_MN%20web.pdf, downloaded and read).
- The current bib URL https://ws.dws.gov.za/IRIS/latestresults.aspx now opens with "It is a great honour to present the Green Drop Report 2025". — [IRIS](https://ws.dws.gov.za/IRIS/latestresults.aspx)
- Wayback copy of the national PDF (CDX status 200, captured 10 Jul 2022): https://web.archive.org/web/20220710190541/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf . An earlier version (30 March 2022) was captured at https://web.archive.org/web/20220402105413/https://ws.dws.gov.za/IRIS/releases/Report_NATIONAL%20_FINAL_30March22_MNEdit_web.pdf . I read the 16 May 2022 final.
- Title: the document calls itself "National Green Drop Report 2022" and "Green Drop Report 2022" (p. 7, index). The bib's "Green Drop National Report 2022" is close enough, but IEEEtranN lower-cases it to "Green drop national report 2022".

### Inferences
- Minimal fix for Ch3:9, keeping the DPW numbers but stating them correctly:
  "The 2022 Green Drop report found that 102 of the 115 wastewater systems run by the Department of Public Works were in a critical state \citep[p.~451]{dws2022greendrop}. The report treats a difference of more than 10\% between the desktop audit score and the site inspection score as a misalignment between administration and the work on the ground, and found differences of up to 59\% \citep[p.~469]{dws2022greendrop}."
  Then change the last sentence, because in these cases the paperwork *understated* the site condition. For example: "...as the report indicates that scores based on submitted paperwork do not always match the condition of the plant."
- Stronger option (rated 8/10): lead with the municipal numbers (334 of 850 municipal systems critical, p. xi) and the municipal TSA/GD deviations of up to 57% (p. 32). These fit a project on municipal Cape Town works better than prison plants do.
- Be careful: the report rates Cape Town's systems as well run (TSA >80%, close GD/TSA match). The motivation should not suggest that the Cape Town plants in the dataset are among the failing ones.
- "59%" is a difference in percentage points between two scores, not "59% of the real state". Writing "59 percentage points" would be more exact.

### Corrected BibTeX
```bibtex
@techreport{dws2022greendrop,
	author      = {{Department of Water and Sanitation}},
	title       = {{National Green Drop Report 2022}},
	type        = {Report},
	institution = {Department of Water and Sanitation},
	address     = {Pretoria, South Africa},
	year        = {2022},
	note        = "\url{https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf}. Archived at \url{https://web.archive.org/web/20220710190541/https://ws.dws.gov.za/IRIS/releases/GD22%20Report_NATIONAL_FINAL_16May22_MN%20web.pdf}. Accessed 16 October 2026",
}
```
(Raw `%20` inside `\url` in a note was test-compiled with the thesis's IEEEtranN setup and printed correctly.)

### Gaps
- I could not see the cover page text (it is an image), so the exact cover title is unconfirmed. The text and index use "Green Drop Report 2022" / "National Green Drop Report 2022".

## eoportal_quickbird

### Takeaway
The page exists and supports "commercial satellite" and "four spectral bands". The thesis sentence is wrong in one detail: the sub-metre resolution (0.61 m) applies only to the single panchromatic band. The four multispectral bands are 2.4 m. Metadata needs small fixes: the page was last updated 8 Jan 2025, and I could not confirm on the page that ESA runs eoPortal.

### Cited Findings
- Thesis Ch3:25: "QuickBird, a commercial satellite that captured sub-metre images in four spectral bands \citep{eoportal_quickbird}."
- eoPortal page: title "QuickBird-2"; last updated 8 Jan 2025; operator DigitalGlobe; launch 18 Oct 2001; 0.61 m panchromatic and 2.4 m multispectral (spec table 0.61–0.72 m PAN, 2.4–2.6 m MS); four MS bands 0.45–0.52, 0.52–0.60, 0.63–0.69, 0.76–0.90 µm. The page contradicts itself on the end date (17 Dec 2015 in quick facts vs re-entry 27 Jan 2015 in the text). — [eoPortal](https://www.eoportal.org/satellite-missions/quickbird-2)
- ESA Earth Online (clearly ESA-published) gives the same: PAN 0.61–0.72 m, MS 2.4–2.6 m, 4 MS bands (blue, green, red, NIR), launched October 2001, operations ended 2015. — [ESA Earth Online](https://earth.esa.int/eogateway/missions/quickbird-2)
- The eoPortal about page does not name an operator. Its contact link goes to an "esatellus" service desk, which suggests ESA but does not prove it. — [eoPortal about](https://www.eoportal.org/about)

### Inferences
- Minimal fix for Ch3:25: "QuickBird, a commercial satellite with a sub-metre panchromatic band and four multispectral bands \citep{eoportal_quickbird}". The four bands matter for the thesis's argument about colour, so this distinction is worth getting right.
- If the author would rather cite an ESA page, the ESA Earth Online page is equivalent and its publisher is unambiguous.

### Corrected BibTeX
```bibtex
@misc{eoportal_quickbird,
	author       = {{eoPortal}},
	title        = {{QuickBird-2}},
	howpublished = {eoPortal satellite missions directory. Last updated 8 January 2025},
	year         = {2025},
	note         = "\url{https://www.eoportal.org/satellite-missions/quickbird-2}. Accessed 6 October 2026",
}
```
Alternative (ESA-published):
```bibtex
@misc{esa_quickbird,
	author       = {{European Space Agency}},
	title        = {{QuickBird-2}},
	howpublished = {ESA Earth Online},
	note         = "\url{https://earth.esa.int/eogateway/missions/quickbird-2}. Accessed 16 October 2026",
}
```

### Gaps
- No Wayback snapshot was confirmed (archive.org rate-limited my requests). I could not find the DigitalGlobe/Maxar QuickBird datasheet at a stable URL.

## gholizadeh2016

### Takeaway
Real, correct metadata. Partially supported. The review covers many parameters (chlorophyll-a, Secchi depth, CDOM, TSS/turbidity, phosphorus, temperature, salinity), so "satellite studies of water quality focus on suspended solids and turbidity" overstates it. The clarity–suspended-solids link is supported. The sentence after the citation ("A clarifier that eliminates solids should therefore look brighter than one that retains them") has the physics backwards: the review says suspended matter causes most of the scattering, and reflectance rises with TSM. The thesis's own Grad-CAM text describes the correctly classified clarifiers as "darker green" and the misclassified one as "pale grey" (Ch6:176).

### Cited Findings
- Crossref: Sensors 16(8), 1298, 16 Aug 2016; Gholizadeh, Melesse, Reddi. The bib's "Gholizadeh, Mohammad Haji" matches how the paper prints the name. — [DOI](https://doi.org/10.3390/s16081298)
- "As water turbidity is mainly the result of the presence of suspended matter, turbidity measurement has often been used to calculate fluvial suspended sediment concentrations and is commonly regarded as the opposite of clarity." — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/)
- "Suspended sediments are responsible for most of the scattering, whereas the absorption is controlled by chl-a and colored dissolved or particulate matter." and "single band algorithms may be adopted where TSM increases with increasing reflectance." — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/)
- Landsat (Ch3:25): "Landsat TM (Thematic Mapper) images have been used extensively due to their relatively low cost, temporal coverage and spatial resolution"; "Landsat-TM is one of the most frequently used sensors to estimate SDD." — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/)
- Small water bodies: the review says this of MERIS/MODIS and hyperspectral sensors ("not many hyperspectral sensors in space with suitable spatial resolution capable of monitoring small lakes and ponds"), not of Landsat. It also notes that IKONOS and airborne sensors were used where more spatial resolution was needed. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5017463/)

### Inferences
- Minimal fix for Ch3:23: "Among the parameters estimated from satellite images, suspended solids and turbidity are the most directly linked to water clarity, since turbidity is mainly caused by suspended matter \citep{gholizadeh2016}. A clarifier that removes solids should therefore look clearer, and usually darker, than one that retains them." This matches both the source and Ch6:176.
- Ch3:25 is supported for "typically use ... Landsat". "Too coarse for small bodies of water" is the author's own inference and is reasonable (30 m pixels against clarifiers of roughly 20–40 m), but it is not a Gholizadeh finding.

### Corrected BibTeX
```bibtex
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
```

### Gaps
- None.

## handannader2019

### Takeaway
Real, correct metadata, and every number in the thesis matches the abstract word for word.

### Cited Findings
- Crossref: Nature Sustainability 2(4), 298–306, 8 Apr 2019. — [DOI](https://doi.org/10.1038/s41893-019-0246-x)
- Abstract: "applying a deep convolutional neural network to high-resolution satellite images ... In North Carolina, the algorithm is able to detect 589 additional poultry CAFOs, representing an increase of 15% from the baseline that was detected through manual enumeration. We show how the approach scales over geography and time, and can inform compliance and monitoring priorities." — [Nature](https://www.nature.com/articles/s41893-019-0246-x)
- Secondary coverage says the imagery was USDA NAIP, which is aerial rather than satellite. The thesis's neutral "high-resolution imagery" avoids that problem. — [Springer Nature community post](https://communities.springernature.com/posts/deep-learning-to-map-concentrated-animal-feeding-operations)

### Inferences
- Ch3:30 is accurate.

### Corrected BibTeX
```bibtex
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
```

### Gaps
- Full text is paywalled. Only the abstract was checked.

## li2022wwtp

### Takeaway
Real, correct metadata, and the claim is supported.

### Cited Findings
- Crossref: IJAEOG 110, 102804, June 2022; Li, Zech, Hong, Ghamisi, Schultz, Zipf. — [DOI](https://doi.org/10.1016/j.jag.2022.102804)
- "We validate our JDL method by training in multiple federal states within Germany and testing in a selected region around Stuttgart." Inputs: VHR images, OpenStreetMap data and Sentinel-2 MSI; 4,187 WWTP features used for training. — [PMC9626640](https://pmc.ncbi.nlm.nih.gov/articles/PMC9626640/)

### Corrected BibTeX
```bibtex
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
```

### Inferences / Gaps
- None.

## lopezbetancur2022

### Takeaway
Real, correct metadata, and the claim is supported. One nuance: the "liquid samples" were distilled water mixed with fish feed or paprika in a 5 cm cube, made for aquaculture. They were not wastewater.

### Cited Findings
- Crossref: Applied Sciences 12(12), 6079, 15 Jun 2022; seven authors as in the bib. — [DOI](https://doi.org/10.3390/app12126079)
- Abstract: "a single image of the liquid sample ... by using a commercial smartphone camera (Android or IOS system) and light-emitting diode (LED) illumination ... We implemented a pre-trained AlexNet model, and an MLR to estimate total suspended solids (TSS), and turbidity ... The best performance was achieved using white light, with an accuracy of 98.24% and 97.20% for TSS and turbidity ... This system was designed for aquaculture environments and tested with both commercial fish feed and paprika." The method kept "constant all parameters (smartphone camera mode, LED lamp intensity, etc.)". — [MDPI PDF](https://mdpi-res.com/d_attachment/applsci/applsci-12-06079/article_deploy/applsci-12-06079.pdf)

### Inferences
- Ch3:23 is fine. Optional: "on smartphone images of prepared liquid samples" makes clear these were lab samples. The 98% is the paper's own accuracy measure for TSS estimation (98.24%, white light).

### Corrected BibTeX
```bibtex
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
```

### Gaps
- None.

## randhawa2023

### Takeaway
Real, correct metadata (UrbanAI '23, pp. 10–21, ACM). "Worked towards worldwide detection" matches the abstract.

### Cited Findings
- Crossref: "Multiscale Multifeature Vision Learning for Scalable and Efficient Wastewater Treatment Plant Detection using Hi-Res Satellite Imagery and OSM", Proc. 1st ACM SIGSPATIAL Int. Workshop on Advances in Urban-AI, pp. 10–21, 13 Nov 2023; authors as in the bib. — [DOI](https://doi.org/10.1145/3615900.3628772)
- Abstract (via OpenAlex): the model "leverages ... Yolov6, RTMDET, EfficientDET, and Domain Adaptation, to accurately and efficiently identify WTP locations worldwide ... We demonstrated the effectiveness of the suggested approach for three diverse global regions: Germany, France, and Malaysia." Gold open access. — [ACM DL](https://dl.acm.org/doi/10.1145/3615900.3628772)

### Corrected BibTeX
```bibtex
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
```

### Inferences
- Optional: "worked towards worldwide detection and tested it in Germany, France and Malaysia" is more concrete.

### Gaps
- Abstract only.

## satoh2021

### Takeaway
Real, correct metadata, and the claim matches the abstract.

### Cited Findings
- Crossref: Environ. Sci.: Water Res. Technol. 7(2), 298–305, 2021; Hisashi Satoh, Yukari Kashimoto, Naoki Takahashi, Takashi Tsujimura. — [DOI](https://doi.org/10.1039/D0EW00908C)
- Abstract (Semantic Scholar): "A deep learning-based two-label classifier 1 recognized a 20% morphological change in the activated flocs. Classifier-2 quantitatively recognized an abundance of filamentous bacteria in activated flocs." A green OA copy is listed at Hokkaido University: https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/83967/2/Satoh%20deep%20learning%20Huscup.pdf (it returned HTML to my script, so I could not read it). — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1039/D0EW00908C)

### Corrected BibTeX
```bibtex
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
```

### Inferences / Gaps
- Abstract only. Ch3:14 ("binary CNN classifiers that recognised changes in floc morphology and the abundance of filamentous bacteria") matches it.

## shuai2023

### Takeaway
Real, correct metadata, and all numbers and the tank-shape remark are in the abstract. One subtlety: the paper's "accuracy" of 80.1% is what it calls AP in Table 3 (137 of 170 detections correct at threshold 0.8), so it is effectively precision.

### Cited Findings
- Crossref: Sustainability 15(7), 5880, 28 Mar 2023; Shuai, Xie, Lu, Chen. — [DOI](https://doi.org/10.3390/su15075880)
- Abstract: "due to the presence of rectangular and circular treatment facilities in sewage treatment plants, the shapes are diverse and the scales are different, resulting in the poor performance of conventional object detection algorithms ... we used a well-trained model to predict the sewage treatment plant from the GF-2 data for the Beijing area ... our method can achieve an accuracy of 80.1% while maintaining the recall rate at a high level (90.4%)." Imagery is 2 m GF-2. Table 3, threshold 0.8: 151 actual plants, 170 detected, TP 137, FP 33, FN 14, AP 80.1%, AR 90.4%. — [MDPI PDF](https://mdpi-res.com/d_attachment/sustainability/sustainability-15-05880/article_deploy/sustainability-15-05880.pdf)

### Inferences
- Ch3:30 is accurate. Optionally write "80.1\% precision at 90.4\% recall" for exactness. Using the paper's own term "accuracy" is defensible.

### Corrected BibTeX
```bibtex
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
```

### Gaps
- None.

## kretzmann2022 (not cited)

### Takeaway
A real GroundUp news article (republished by Daily Maverick, 21 June 2022, by Steve Kretzmann) about the 2022 Green Drop report. It repeats the national figure (334 of 850 municipal works critical) and argues that some Green Drop award winners still pollute: City of Cape Town's Green Point and Hout Bay marine outfalls hold awards but are exempt from monitoring ("No Monitoring Required"). Recommendation: delete it, or cite it as secondary support for "a score can hide true performance" (rated 5/10). Any numbers should come from the DWS report itself.

### Cited Findings
- og:title "'A tsunami of human waste' - Half of SA's sewage treatment works are failing, says report"; author Steve Kretzmann; published 2022-06-21; tagged GROUNDUP. The URL slug ends "says-study", the headline says "says report". The bib title matches the headline. — [Daily Maverick](https://www.dailymaverick.co.za/article/2022-06-21-a-tsunami-of-human-waste-half-of-sas-sewage-treatment-works-are-failing-says-study/)
- "Of 850 municipal wastewater treatment works, 334 (39%) are in a critical state"; "Five of the 22 Green Drop award winners fall into this category [No Monitoring Required]. All of these are in the Western Cape"; "The Green Point and Hout Bay wastewater treatment works in the City of Cape Town are both Green Drop award winners. They are also marine outfalls." — same

### Inferences
- The only sentence where it would help is Ch3:9's closing claim that scores "can hide true performance". Kretzmann actually supports that direction (a high score despite pollution), unlike the DPW numbers. It is journalism, though, and the Cape Town outfalls are not activated sludge plants. Better to cite DWS directly and drop this entry. If kept, cite it as "\citet{kretzmann2022} notes that..." and do not use it for statistics.

### Gaps
- I did not locate the GroundUp original URL.

## westbrook2016 (not cited)

### Takeaway
Real: a student essay (first prize, 2016 ASCE Student Competition, undergraduate) on the Association for the Study of the Cuban Economy site, dated 30 Nov 2016. It is the **closest precedent to the thesis**. The authors viewed six Havana-area WWTPs in Google Earth Pro historical imagery and judged visually whether units were in service over time, for example "the WWTP was operational, as identified by water in both circular clarifiers" and "In 2002 none of the six circular units appeared to be in service". It should be cited, not deleted (rated 8/10). Citing it also makes Ch3:30 more honest.

### Cited Findings
- Authors Alexandra Westbrook and Nayara Sabrina de Freitas Alves; "Editor's Note: This essay was awarded First Prize in the 2016 ASCE Student Competition for undergraduate students." Acknowledges Dr Helena Solo-Gabriele (University of Miami). — [ASCE Cuba database](https://ascecubadatabase.org/asce_proceedings/havanas-wastewater-treatment-plants-changes-over-time-and-estimate-of-replacement-cost/)
- Method: "four of Havana's and two of Havana Province's Wastewater Treatment Plants (WWTPs) were observed via satellite imagery Google Earth Pro and compared to images from previous years in order to determine if any changes could be identified visually." — same
- Example judgements: "By September 1, 2009, the WWTP was operational, as identified by water in both circular clarifiers"; "In 2002 none of the six circular units appeared to be in service. In 2006 and as recently as in the 2015 image, three out of the six units appeared to be back in service." — same

### Inferences
- Where to cite it: Ch3:30 says "In all of this work the tanks help to find a plant, and none of these studies asks whether the tanks are working." Westbrook did ask that, manually and qualitatively. A suggested addition after that sentence: "The only attempt found to judge from such imagery whether units were in service is a manual comparison of historical Google Earth images of six plants around Havana \citep{westbrook2016}, which inferred, for example, that a plant was operating because its clarifiers held water." It also fits Ch1:4 ("manual analysis of aerial photography by experienced personnel") and supports using historical Google Earth imagery (Ch1:15, Ch4:4).
- It is grey literature (an undergraduate prize essay) and should be described as such. It is still a fair, honest precedent.

### Corrected BibTeX
```bibtex
@misc{westbrook2016,
	author       = {Westbrook, Alexandra and {de Freitas Alves}, Nayara Sabrina},
	title        = {Havana's wastewater treatment plants: Changes over time and estimate of replacement cost},
	howpublished = {Association for the Study of the Cuban Economy (ASCE). First prize, 2016 ASCE Student Competition for undergraduate students},
	year         = {2016},
	note         = "\url{https://ascecubadatabase.org/asce_proceedings/havanas-wastewater-treatment-plants-changes-over-time-and-estimate-of-replacement-cost/}. Accessed 30 September 2026",
}
```
(The existing entry is correct. No change is needed beyond citing it.)

### Gaps
- The essay probably also appears in ASCE's "Cuba in Transition" proceedings volume for 2016 (likely vol. 26). I did not verify this, so it is not added to the entry.
- No Wayback snapshot was confirmed for this page or the Daily Maverick article (archive.org rate-limited me).
