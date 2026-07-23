# ----------------------------------------------------------------------------- #
  Annotations created using VIA2 (https://www.robots.ox.ac.uk/~vgg/software/via/)
# ----------------------------------------------------------------------------- #

Fine Bubble Diffused Aeration (FBDA) wastewater treatment works (WWTW) included in the data set: 

  * Atlantis
  * Cape Flats
  * Waterval
  
* All images were sourced from Google Earth.
* Dates in filenames correspond to the date on which the image was captured.
  Naming convention: month_year (xx_yyyy), e.g. 03_2024 -> March 2024.
  Waterval 07_2016 (2) -> (2) indicates that there were two images available for the relevant month.
  
# ----------------------------------------------------------------------------- #
All plants have biological reactors. Each reactor consists of an aeration zone and one or more clarifiers.

* Aerobic Zone Attributes:
  1) "functional":"leopard print"
  2) "dysfunctional":"two thirds or more of surface has no leopard print or clearly dysfunctional"
  3) "suboptimal":"a third of surface or more has no leopard print or unsure"

* Clarifier attributes:
  1) "Dysfunctional":"Green"
  2) "Functional":"Black"
  3) "Scum":"Surface not uniformly black"
  4) "Empty":"Shadows visible inside tank"

# ----------------------------------------------------------------------------- #
* Atlantis
  
  - Images (48)
  - Labels
	> Atlantis Aerobic Zone.json (rectangles)
	> Atlantis Clarifiers.json (circles)
  - AtlantisOverview -> Overview of the plant, no labels assigned
	
# ----------------------------------------------------------------------------- #
* CapeFlats

  - Images
    > CF1 -> Images of unit 1 (96)
    > CF2 -> Images of unit 2 (94)
    > CF3 -> Images of unit 3 (92)
    > CF4 -> Images of unit 4 (100)
  - Labels
	> Cape Flats 1 Clarifiers.json (circles)
	> Cape Flats 2 Clarifiers.json (circles)
	> Cape Flats 3 Clarifiers.json (circles)
	> Cape Flats 4 Clarifiers.json (circles)
	> Cape Flats 1 Aerobic Zones.json (rectangles)
	> Cape Flats 2 Aerobic Zones.json (rectangles)
	> Cape Flats 3 Aerobic Zones.json (rectangles)
	> Cape Flats 4 Aerobic Zones.json (rectangles)
  - CapeFlatsOverview03_2024 -> Overview of the plant, no labels assigned
  - Unit_Numbers_CapeFlatsOverview03_2024 -> Numbers assigned to biological reactor units
	
# ----------------------------------------------------------------------------- #
* Waterval

  - Images 
    > Waterval (Overview) -> Images of entire of the plant, no labels assigned
    > Top -> Images of top unit (37)
    > Bottom -> Images of bottom unit (36)
  - Labels
	> Waterval Aerobic Zone Bottom.json (rectangles)
	> Waterval Aerobic Zone Top.json (rectangles)
	> Waterval Clarifiers Bottom.json (circles)
	> Waterval Clarifiers Top.json (circles)
	
# ----------------------------------------------------------------------------- #

