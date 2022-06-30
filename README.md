# NewsLDA: Nachrichtenanalyse mittels Latent Dirichlet Allocation

Pfadangaben müssen angepasst werden.

Progammcode findet sich im Ordner "scripts":
- Vorverarbeitung der Daten: run_preprocessing.py
- LDA-Modell mit Gensim: run_model.py
- Funktionen zur Evaluierung und Visualisierung: eval_and_vis.py
- Funktionen zur thematischen Analyse: analyze.py

Daten für die Analyse finden sich im Ordner "data":
- modeldata: gespeicherte LDA-Modelle
- sueddeutsche, taggesspiegel, wirtschaftswoche: monatliche Artikel als CSV-Datei
- all_articles.csv u.ä.: verwendete Korpora

Das für das LDA-Modell verwendete Korpus "all_articles.csv" kann aufgrund seiner Größe nur in komprimierter Form hochgeladen werden. Es kann auch rekreiert werden, indem merge_csv.py auf den csv-Dateien "sueddeusche/ww/tagesspiegel_all.csv" ausgeführt wird.
