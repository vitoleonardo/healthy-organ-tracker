*Viktor Krawutschke 2542058 Proposal Bachelorarbeit*

**Segmentation of the Gastrointestinal Tract based on MRI Images using Deep Learning / Segmentierung des Magen-Darm-Traks anhand von MRT Bildern mithilfe von Deep Learning**

## Motivation

Jedes Jahr wird bei millionen von Menschen Krebs im Verdauungstrackt festgestellt (Gastrointestinale Stromatumoren kurz GIST), häufig verläuft dieser tödlich. Eine Behandlungsmethode ist hierbei die Strahlentherapie, bei der gezielt die befallenen Zellen bestrahlt werden. Die genaue Position des Tumors sowie den restlichen Eingeweiden unterscheidet sich jedoch und muss täglich neu bestimmt werden. Der behandelnde Arzt macht sich ein Bild von den Organen des Patientes mithilfe einer Magnetresonanztomographie und muss händisch den Tumor lokalisieren, was in manchen Fällen bis zu 60 Minuten dauert. Die Idee ist es mithilfe des Datensatzes des UW-Madison Carbone Cancer Center und Deep Learning ein Model zu entwerfen, welches automatisch die Segmentierung der Organe übernimmt. Das würde Ärzten und Patienten in zukunft eine schnellere und genauere Behandlung erlauben.

## Daten
### train.csv

| Feld      | Datentyp | Beschreibung                |
| ---------- | --------- | -------------------------- |
| id | number    | Einzigartige ID für jedes Objekt |
| class    | string    | string Repräsentation ein jeder Klasse: large\_bowel, small\_bowel und stomach    |
| segmentation      | number    | RLE encodierte Pixel für das identifizierte Objekt       |

### train Ordner
Ordner mit Fall/Tag Ordnern gefüllt mit Teilbildern (slice\_0001, slice\_0002...) für jeden Case  an einem bestimmten Tag. Jedes Slice hat 4 Zahlen im Namen die zum einen die Höhe und Breite des Slices an sich repräsentieren sowie die Pixeldichte in Milimetern. Ein Tag besteht aus 144 bzw. in seltenen Fällen 80 Slices. Jeder Case hat zwischen 1 und 5 Tagen Bildmaterial, demnach stehen für jeden Case 144 - 720 Bilder zur verfügung.

## Aufgabenstellung
#### Datenaufbereitung
* Extrahieren von Datenpunkten wie case\_id, tag\_xy sowie denen im Namen enthaltenen Informationen.

### State of the Art Modelle im Bereich Segmentierung
Anwenden von geeigneten SOTA Modellen im Rahmen eines Python Programmes die sich mit Bildsegmentierung beschäftigen wie z.B. HRNetV2, DeepLabV3 oder ähnliche. 

### Auswertung
Ziel ist es einen möglichst hohen Platz im Ranking im Rahmen der [Kaggle Competition] (https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/) zu erreichen. Der Score wird mithilfe einer submission.csv, welche aus id, class und predicted besteht. Gemessen wird zu 40% aus dem [Dice coefficient](https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient) und 60% der [3d Hausdorff distance] (https://en.wikipedia.org/wiki/Hausdorff_distance).  

&nbsp;
&nbsp;

## Mindestanforderungen

### Recherche
Intensive Recherche zu den Themen GIST, Datenaufbereitung sowie aktuellen Segmentierungsarchitekturen.

### Training 
Anwenden von drei ausgesuchten Architekturen mithilfe von bekannten Machine Learning Frameworks wie Keras, Tensorflow und/oder Scikit-learn sowie Dokumentierung der Performance.

### GIT
Bereitstellen des Codes in einem dafür angelegtem Repository inkl. einer ausführlichen README.md, die die Themen Installation, Training, Prediction und Visualisierung behandelt. 
