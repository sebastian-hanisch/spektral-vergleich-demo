# Welches Verfahren, gemessen woran? Der Sieger steht im Szenarienmix – der Reuegrad zeigt, was robust ist – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-spektral-vergleich-demo.streamlit.app/)**

Sechzehnte und letzte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie baut auf den Verfahren der Demos 1 bis 15 auf
(Periodogramm, Multitaper, AR-Spektrum, Welch, Unterraumverfahren, Frequenzschätzer, Peak-oder-Rauschen-Tests, Lomb-Scargle, Spektrogramm, Neuzuordnung, Wavelets, Wavelet-Schrumpfung) und führt **keine neuen Verfahren** ein.
Sie misst sie an **sieben Aufgaben mit bekannter Wahrheit** gegeneinander und fragt, was ein Vergleich eigentlich entscheidet: Der Sieger hängt am **Szenarienmix**, und der **Reuegrad** (englisch *regret*) zeigt, was robust ist.

## Warum dieses Problem

„Welches Verfahren ist das beste?“ hat keine Antwort ohne ein „wofür“. In dieser Tabelle gewinnt das AR-Spektrum jedes Szenario mit autoregressivem Hintergrund und verliert bei der Kerbe in allen drei Längen (n = 256 und 1024 gegen Welch, n = 64 gegen Multitaper und Welch); der
Fisher-Test auf weißes Rauschen arbeitet in weißem Rauschen (schwächer als mit AR(2)-Vorfilterung) und ist in AR(1)-Rauschen nutzlos; Synchrosqueezing liegt beim Chirp mit 10 dB gleichauf mit dem Spektrogramm und bricht bei 0 dB ein (bei der Sinus-FM schon bei 10 dB). Wer Verfahren vergleicht, wählt Szenarien,
und die Wahl entscheidet mit. Die Demo macht das messbar: gleiche Szenarien für alle, ein Maß (der Reuegrad: wie viel schlechter als das beste Verfahren im Szenario), und ein Regler, der den Mix der Szenarien verschiebt.

## Modell

Sieben Aufgaben (alle simuliert, die Wahrheit ist bekannt):

- **A Spektrum schätzen:** sieben Hintergründe (Resonanz, Resonanz plus weißes Rauschen, Steil, zwei scharfe Resonanzen, weiß; dazu Kerbe und Bandpass, die kein autoregressives Spektrum haben) bei n = 64, 256 und 1024;
  Verfahren: Hann-Periodogramm, Welch (Hann, 50 %, Segment n/4), Multitaper (NW = 4), AR-Spektrum (Burg, Ordnung nach BIC); Fehler = quadratischer Mittelwert der Differenz der Logspektren in dB.
- **B Zwei Linien trennen:** zwei gleich starke Linien bei 0,2 im Abstand von 0,75 bis 2,5 Bins (n = 64) mit 20 oder 10 dB je Linie; Anteil getrennter Läufe für Hann, Multitaper (NW = 2), AR (Ordnung 16 und BIC), MUSIC und ESPRIT (beide mit vorgegebener wahrer Linienzahl 4) sowie ESPRIT mit per MDL geschätzter Linienzahl.
- **C Eine Linie genau treffen:** Frequenzfehler als Vielfaches der Cramér-Rao-Schranke für Periodogrammspitze, Jacobsen, Candan, Hann-Spitze, MUSIC und ESPRIT (beide mit vorgegebener Linienzahl 2) sowie AR-Spitze.
- **D Linie im Rauschen finden:** weißes und AR(1)-Rauschen (Pol 0,8), n = 128; Fisher-Test, Fisher-Test nach AR(2)-Vorfilterung und harmonischer F-Test, **auf wirklich 5 % Falschalarm kalibriert** (Schwelle aus 400 Rauschläufen), daneben die Raten mit der Theorieschwelle.
- **E Unregelmäßige Messzeiten:** Lomb-Scargle für vier Muster der Messzeiten (regelmäßig, Jitter, zufällig, Nachtlücken), Linien bei der 2,6fachen und der 0,4fachen mittleren Nyquist-Frequenz; unter der Nyquist-Grenze dazu das Interpolieren aufs Gitter.
- **F Zeitabhängige Frequenz:** Chirp und Sinus-FM, rauschfrei und mit Rauschen; Frequenzfehler von Spektrogramm, Neuzuordnung, Synchrosqueezing, CWT-Kamm und geglätteter Phasenableitung.
- **G Entrauschen:** Blocks, Bumps, HeaviSine, Doppler (n = 1024); Wavelet mit SURE-Schwelle, Wavelet mit universaler Schwelle und Fourier mit universaler Schwelle; dazu Orakel-Bezüge (beste Fourier-Schwelle, bester gleitender Mittelwert).

**Reuegrad:** bei Fehlern Fehler / bester Fehler im Szenario (1 = bester), bei Erfolgsanteilen die Punkte hinter dem besten. Je Verfahren das Mittel (mit den Gewichten des Szenarienmixes), das Maximum (Minimax-Kriterium) und die Zahl
der Siege. Die **Entscheidungshilfe** empfiehlt bei bekanntem Typ den Sieger des nächstgelegenen Szenarios, bei unbekanntem das Verfahren mit dem kleinsten größten Reuegrad. Der Kern ist in numpy geschrieben; scipy dient nur als
Gegenprobe in den Tests. Die Verfahren sind aus den Vorgängerdemos übernommen, bis auf Modulbezüge und Standardwerte (`spv_estimators.py`, `spv_timefreq.py`, `spv_denoise.py`).

## Methodik

Die Tabelle der Experimente ist **vorgerechnet** (`spv_results.json`, 63 Szenarien, erzeugt mit `python spv_results.py`) und deterministisch mit festen Zufallsströmen: 100 Läufe je Szenario bei A, 200 bei B, 300 bei C, 300 bei D
(dazu 400 Rauschläufe für die Schwellen), 300 bei E, 20 bei F (verrauscht) und 40 bei G. Die Live-Ansicht rechnet eine einzelne Realisierung (Seed einstellbar); die Tests rechnen Stichproben der Tabelle nach. „Wahr“ ist
der Wert, mit dem die Reihen gebaut sind.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wer gewinnt das Spektrum (*Spektrum schätzen und der Szenarienmix*)? | Das AR-Spektrum (Burg, BIC) gewinnt 15 von 15 Szenarien mit autoregressivem oder weißem Hintergrund, aber nur 1 von 6 mit Kerbe oder Bandpass. Steil, n = 256: AR 0,78 dB, Welch 2,91 dB, Multitaper 4,01 dB, Hann-Periodogramm 6,04 dB. Kerbe, n = 1024: Welch 1,57 dB, Multitaper 1,73 dB, AR 2,09 dB. Bandpass, n = 256: Hann-Periodogramm 7,01 dB, AR 9,54 dB, Welch 16,91 dB, Multitaper 35,77 dB. |
| Was zeigt der Reuegrad (gleiches Experiment)? | Größter Reuegrad über alle 21 Szenarien: AR 1,91, Multitaper 7,96, Welch 8,85, Hann-Periodogramm 28,31 (ein Quotient: Der Wert des Hann-Periodogramms entsteht bei weißem Hintergrund und n = 1024, wo der beste Fehler nur 0,22 dB beträgt). Unter dem Mix 71 % autoregressiv ist der mittlere Reuegrad 1,13 (AR), 3,80 (Welch), 4,18 (Multitaper) und 7,70 (Hann-Periodogramm); bei 0 % autoregressiv 1,43, 1,96, 3,29 und 2,09. Nach Siegen kippt es dagegen: Das AR-Spektrum liegt erst ab einem autoregressiven Anteil von 26 % vor Welch. |
| Welches Verfahren trennt zwei Linien (*Linien trennen und genau treffen*)? | Bei einem Bin Abstand und 10 dB je Linie (n = 64) trennt ESPRIT 100,0 % der Läufe, MUSIC 96,5 %, das AR-Spektrum mit Ordnung 16 nur 45,0 %, das Hann-Periodogramm 12,5 %, das AR-Spektrum mit BIC-Ordnung 0,5 % und das Multitaper 0,0 %. ESPRIT und MUSIC bekommen dabei die wahre Linienzahl (4) vorgegeben; ESPRIT mit per MDL geschätzter Linienzahl trennt in dieser Zelle 93,5 %. ESPRIT (Linienzahl vorgegeben) trennt in allen acht Szenarien mindestens 94,0 % und hat überall den Reuegrad null (mit MDL-Zählung mindestens 85,0 % und größter Reuegrad 9); der größte Reuegrad von MUSIC ist 44 Punkte, der des AR-Spektrums mit Ordnung 16 80. |
| Welches Verfahren trifft eine Linie? | Bei n = 64 und 10 dB liegt der Fehler der Spitze des Rechteck-Periodogramms bei 1,03 Cramér-Rao-Schranken, bei ESPRIT 1,06, MUSIC 1,18, Jacobsen und Candan 1,53 und der AR-Spitze 4,48. Mittlerer Reuegrad der Spitze 1,10, größter 1,49; die AR-Spitze hat 10,17 und 22,37. Der Abstand zwischen Spitze (1,03) und ESPRIT (1,06) liegt innerhalb der Stichprobenstreuung (300 Läufe, Standardfehler eines RMSE etwa 4 %). |
| Was kostet ein falsches Rauschmodell (*Eine Linie im Rauschen finden*)? | Bei AR(1)-Hintergrund schlägt der Fisher-Test auf weißes Rauschen in 99,2 % der reinen Rauschläufe an; auf wirklich 5 % kalibriert findet er eine Linie von −12 dB in 5 %, mit AR(2)-Vorfilterung in 72 % und der harmonische F-Test in 38 % der Läufe. Bei weißem Hintergrund und −6 dB sind es 68 %, 82 % und 52 %. Der F-Test mit Theorieschwelle (Bonferroni über 64 Frequenzen) hat schon bei weißem Rauschen 14 % und bei AR(1)-Hintergrund 15 % Falschalarm statt 5 %; der Überschuss liegt an der Schwelle bei dichtem Frequenzraster, nicht am gefärbten Rauschen. Die Vorfilterung ist in allen zehn Szenarien Sieger oder gleichauf. |
| Was leistet Lomb-Scargle (*Unregelmäßige Messzeiten und zeitabhängige Frequenz*)? | Bei einer Linie über der Nyquist-Grenze (0 dB) erkennen zufällige Messzeiten 100 % der Läufe, regelmäßige 26 %, Jitter 100 %, Nachtlücken 92 %. Unter der Grenze verliert das Interpolieren aufs Gitter: 65 % gegen 100 % bei 0 dB und 5 % gegen 47 % bei −6 dB. |
| Was leisten die Zeit-Frequenz-Verfahren? | Beim Chirp mit 10 dB liegen Neuzuordnung (0,00049), Spektrogramm (0,00057) und Synchrosqueezing (0,00059) vor dem CWT-Kamm (0,00103); bei 0 dB bricht Synchrosqueezing ein (0,0521 gegen 0,0037). Bei der Sinus-FM (20 dB) liegen die Phasenableitung (0,0021) und der CWT-Kamm (0,0035) vor dem Spektrogramm (0,0046). Mittlerer Reuegrad: Neuzuordnung 1,59 (größter 2,54), Phasenableitung 2,59 (6,83), Synchrosqueezing 11,11 (39,12, in der rauschfreien Chirp-Zelle vom Frequenzraster bestimmt). |
| Wer entrauscht (*Entrauschen*)? | Wavelet-SURE gewinnt alle vier Signale (Blocks 0,37, Bumps 0,39, HeaviSine 0,10, Doppler 0,23) vor der Fourier-Schwelle (1,25, 1,35, 0,19, 0,78); bei Blocks und Bumps ist selbst die beste Fourier-Schwelle (Orakel) mit 0,97 und 1,00 kaum besser als die verrauschte Reihe (0,99 und 1,01). |

## Ehrliche Grenzen

- **Die Szenarien sind gewählt, nicht repräsentativ:** Mit anderen Hintergründen, Längen und Rauschabständen kippt der Sieger. Der Reuegrad gilt nur für diese Zellen; die Entscheidungshilfe empfiehlt nur innerhalb der Tabelle.
- **Heimvorteil in der Auswahl:** Fünf der sieben Hintergründe der Aufgabe A sind autoregressiv oder weiß, wo das AR-Spektrum zu Hause ist. Der Regler für den Mix zeigt, wie viel das ausmacht; nach dem mittleren Reuegrad bleibt
  das AR-Spektrum auch bei einem Mix ganz ohne autoregressive Hintergründe vorn (1,43 gegen 1,96 bei Welch), nach Siegen nicht.
- **Feste Einstellungen:** Jedes Verfahren läuft mit einer festen Einstellung (Fenster, Ordnung, Taper, Schwellen). Mit besserer Einstellung kippt mancher Abstand; die Demo vergleicht Standardeinstellungen, keine Höchstleistung. MUSIC und ESPRIT bekommen die wahre Linienzahl vorgegeben (Aufgaben B und C); ESPRIT mit MDL-Zählung ist die Variante ohne.
- **Nur Simulation:** Die Wahrheit ist bekannt, die Modelle der Szenarien sind einfach (weißes oder AR(1)-Rauschen, eine oder zwei Linien, zwei Testsignale). Mit echten Daten fehlt der Vergleichswert.
- **Ein Lauf ist kein Mittel:** Die Live-Ansicht zeigt eine Realisierung, die vom Mittel abweichen kann; die Tabelle mittelt 20 bis 400 Läufe, ihre Werte haben Stichprobenfehler (bei Anteilen mit 300 Läufen höchstens 2,9 Punkte Standardfehler, bei Aufgabe B mit 200 Läufen 3,5; die Schwellen stammen aus 400 Rauschläufen).
- **Das Interpolieren kennt nur Linien unter der Nyquist-Grenze:** In den Szenarien darüber fehlt es in der Tabelle; sein Reuegrad stammt aus den drei Szenarien darunter.
- **Quellen** (Savage 1951 zum Reuegrad; die Quellen der einzelnen Verfahren stehen in den Demos 1 bis 15) wurden nicht gelesen; die Maße sind per Handrechnung und Simulation geprüft.

## Verifikation

- **Kern:** Reuegrad, Mittel, Maximum, Siege, Minimax und Szenarienmix gegen Handrechnung an Mini-Zellen; Entscheidungshilfe (Abstandsregeln, Empfehlung, Hinweis); Szenarien (Kerbe, Bandpass, AR-Hintergrund, Linien, Rauschen) gegen
  Handrechnung; Aufgaben mit Schätzern gegen scipy, feste Werte kleiner Läufe und Handfälle (Cramér-Rao-Schranke, Schwellen, Fensterbereich, Fehlermaß); Stichproben der vorgerechneten Tabelle werden neu gerechnet.
- **Übernommene Verfahren (bis auf Modulbezüge und Standardwerte):** Die aus den Vorgängerdemos übernommenen Funktionen sind mit den Quellen verglichen (`tests/test_extracted_modules.py`, wo die Quellrepos vorliegen) und dort einzeln geprüft.
- **Experimente:** feste Werte (`tests/test_experiments_golden.py`), jede Zahl dieser README und der Texte in der App an die Tabelle gebunden (`tests/test_claims.py`), Verdikte an handgesetzten Kennzahlen.
- **Mutationstest:** `tools/mutate_demo.py` über Aufgaben, Reuegrad, Entscheidungshilfe, Auswertung und Tabellenerzeugung (`spv_tasks`, `spv_regret`, `spv_advisor`, `spv_evaluation`, `spv_results`): alle 339 Stellen sind gefangen oder als äquivalent festgehalten (49, meist `>` gegen `>=` bei stetigen Werten und Konstanten außerhalb des genutzten Bereichs; Zustand in `tests/mutation_state.json`, die Datei enthält durch frühere Fassungen geänderter Zeilen mehr Einträge). Nicht mutiert sind die wortgleich übernommenen Module (eigene Tests in den Quellen), die Szenario-Erzeugung `spv_scenarios.py` (durch Handrechnung und feste Werte gesichert), Darstellung und App-Glue.
- **Presets, Oberfläche:** Vollständigkeit und Grenzen, AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Permalink, bedingte Regler, Figuren, Footer, Laufzeit-Importe.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html).
