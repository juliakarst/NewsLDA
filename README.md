# NewsLDA: Nachrichtenanalyse mittels Latent Dirichlet Allocation

Pfadangaben müssen vor Nutzung angepasst werden!

Progammcode findet sich im Ordner "scripts":
- Vorverarbeitung der Daten: run_preprocessing.py
- LDA-Modell mit Gensim: run_model.py
- Funktionen zur Evaluierung und Visualisierung: eval_and_vis.py
- Funktionen zur thematischen Analyse: analyze.py

Daten für die Analyse finden sich im Ordner "data":
- modeldata: gespeicherte LDA-Modelle
- monatliche Artikel einer Zeitung als CSV-Datei im jeweiligen Ordner
- all_articles.csv u.ä.: verwendete Korpora

Die Ordner "Grafiken" und "Daten-Analyse" enthalten Statistiken und Tabellen zur Auswertung und Interpretation.

Das für das LDA-Modell verwendete Korpus "all_articles.csv" kann aufgrund seiner Größe nur in komprimierter Form hochgeladen werden. Es kann auch rekreiert werden, indem merge_csv.py auf den csv-Dateien "sueddeusche/ww/tagesspiegel_all.csv" ausgeführt wird.
