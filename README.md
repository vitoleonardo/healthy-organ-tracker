# Healthy Organ Tracker

UW-Madison GI Tract Image Segmentation Challenge - BA - Viktor Krawutschke

Bei diesem Repository handelt es sich um Basis, auf die die Bachelorarbeit mit dem Namen "Segmentierung des Magen-Darm-Trakts anhand von MRT Bildern mithilfe von Deep Learning" aufbaut.

### HOT Data Exploration

In diesem Notebook wird sich allgemein mit dem Datensatz auseinandergesetzt.

### HOT Performance

Dieses Notebook beinhaltet alle Funktionen, zum Erfassen der Performance der Modelle. Die Daten werden aus der Historie der Metriken der Modelle erhoben.

### HOT Inference

Dieses Notebook enthält alle Funktionen zum Visualisieren der trainierten Modelle.

### hpc_train_files

In diesem Ordner befinden sich die einzelnen Klassen die zum Trainieren auf dem Hilber-Cluster genutzt wurden. Benutzung:

```
HOT.py <backbone> <dim> <batch> <epochs> <semi3d_data> <remove_faulty_cases> <use_crop_data> <selected_fold>
```

### input

Hier befinden sich verschiedene Versionen des Tranings-Dataframes.

### jobscript

Skript, welches benutzt wurde um Aufträge in die Queue zu schicken.

### Latex

Latex Dateien der Thesis.

### Performance

Hier finden sich die Werte der Metriken der Predictions der einzelnen Modelle.

### tensorboard_logs

Hier befinden sich die Werte der Trainingsverläufe sowie die Dateien für Tensorboard.
