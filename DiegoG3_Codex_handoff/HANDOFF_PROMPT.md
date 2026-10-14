# Pierwszy prompt do Codexa w VS Code

Kontynuujemy projekt reverse engineeringu **Diego G3 3.0.8.2 ECU**. Nie masz dostępu do wcześniejszej rozmowy ChatGPT; pełne dotychczasowe ustalenia są w tym repozytorium.

1. Przeczytaj **AGENTS.md**, **PROTOCOL.md**, **REVERSE_ENGINEERING.md**, **TASKS.md** oraz katalog `reference/`.
2. Uruchom **`python -m unittest discover -s tests -v`** i sprawdź działanie `src/diego_protocol.py`. Podsumuj stan: co jest potwierdzone w EXE [X], co zmierzone w GUI [M], co pochodzi tylko z emulatora [E] oraz hipotezy [H].
3. Oceń architekturę referencyjnego dekodera i zaproponuj zwięzły plan rozwoju klienta diagnostycznego **wyłącznie do odczytu**.
4. W następnym kroku zajmiemy się **bajtem 45 z ramki 0B**, jego bramką capability w `0x0056409C` i wyjaśnieniem, dlaczego GUI nie reagowało na testowane bity. Nie wprowadzaj niepotwierdzonych interpretacji ani nowych transmisji do rzeczywistego ECU.
5. Kiedy odkryjesz coś nowego, dokumentuj to w `FINDINGS.md`, dodaj test regresyjny i aktualizuj obie specyfikacje. Nie zmieniaj sprawdzonych algorytmów RPM/MAP/LUT/czasów wtrysku bez dowodu.

Proszę najpierw o przeczytanie i przetestowanie projektu, a następnie krótkie sprawozdanie gotowości oraz propozycję **pierwszego konkretnego eksperymentu**.
