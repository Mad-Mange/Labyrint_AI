# Labyrint AI

En digital version av det klassiska trälabyrintspelet från 70/80-talet (se
`pictures/Labyrint.png`). Med två rattar lutar man brädet och försöker föra stålkulan
från **START** till **FINISH** utan att den ramlar ner i något av de 23 numrerade hålen.

Spelet är byggt för att ett neuralt nätverk ska kunna lära sig spela det: fysikmotorn
körs helt utan grafik (ca 700 gånger snabbare än realtid) och har ett färdigt
[Gymnasium](https://gymnasium.farama.org/)-API, standarden för förstärkningsinlärning.

## Kom igång

```powershell
# en gång: skapa miljön (Python 3.12) och installera beroenden
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# spela
.venv\Scripts\python play.py
.venv\Scripts\python play.py --a             # titta på den inbyggda autopiloten
.venv\Scripts\python play.py --ai            # titta på den tränade AI:n (eller dubbelklicka spela_ai.bat)
```

Kör alltid med `.venv\Scripts\python` – vanliga `python` har inte pygame installerat.

| Kontroll | Funktion |
|---|---|
| Mus | Luta brädet (musens avstånd från mitten = lutning) |
| Pilar / WASD | Vrid rattarna (lutningen ligger kvar när du släpper, som på riktigt) |
| Mellanslag | Plant bräde |
| R | Börja om |
| P | Paus |
| F1 | Visa vad AI:n "ser" (sensorer) |
| F2 | Växla mus → autopilot → AI |
| Esc | Avsluta |

## Struktur

```
labyrint/
  geometry.py      väggar (roterbara rektanglar), den svarta linjen (Path), rutnät
  level.py         laddar och validerar banor från levels/*.json
  levels/classic.json
  game.py          fysikmotorn – LabyrinthGame (ingen grafik!)
  render.py        pygame-grafik (läser bara spelet, ändrar det aldrig)
  autopilot.py     handskriven regulator som följer linjen – riktmärke för AI:n
  env.py           Gymnasium-miljön "Labyrint-v0" – API:t för AI:n
  agent.py         låter en tränad modell styra spelet (NeuralPilot)
  vecenv.py        kör många miljöer per process – snabbare träning
models/            färdigtränad AI (labyrint_ai.zip)
play.py            spela själv
train.py           träna AI:n
dashboard.py       grafer över en träning, live
spela_ai.bat       dubbelklicka: titta på den tränade AI:n
trana_ai_live.bat  dubbelklicka: träna en ny AI och se den lära sig live
tests/             pytest
```

**Fysik:** kulan rullar med a = 5/7 · g · sin(lutning) (massiv kula som rullar utan att
glida), max lutning 3°, brädet vrids med begränsad hastighet (som rattarna), rullmotstånd,
studs mot väggarna och en realistisk hålkant: när kulans mittpunkt passerar kanten tippar
den in mot hålet. En snabb kula kan alltså sladda förbi ett hål, en långsam ramlar i.

## API för AI-delen

### Direkt mot motorn

```python
from labyrint import LabyrinthGame, Status

game = LabyrinthGame("classic")
game.reset()                         # eller reset(start_s=500) för att starta mitt på banan
while game.status is Status.RUNNING:
    game.step((0.2, -0.8))           # mållutning x, y i [-1, 1]; ett steg = 1/60 s
print(game.state)                    # GameState: position, fart, lutning, framsteg, hål ...

snapshot = game.clone()              # billig kopia – för sökning/planering
```

### Gymnasium (för träning)

```python
import gymnasium as gym
import labyrint                      # registrerar "Labyrint-v0"

env = gym.make("Labyrint-v0", render_mode="human")   # eller None vid träning
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- **Handling:** 2 tal i [-1, 1] – brädets mållutning (x, y), 30 beslut per sekund.
- **Observation:** 36 tal i [-1, 1]: kulans position/fart, lutning, linjen 15/40/80 mm
  framåt, linjens riktning och avstånd till linjen, de 4 närmaste hålen,
  avstånd till väggar i 8 riktningar och hur långt man kommit. (Tryck F1 i spelet så ser du dem.)
- **Belöning:** +0,1 per mm framåt längs linjen (bakåt ger minus), −5 för hål,
  +50 för mål, −0,01 per steg. Justeras via `RewardConfig`.
- **Ingen parkering:** står kulan still (inga nya framsteg på 10 s, `stall_seconds`)
  räknas det som att den ramlat. Utan den regeln lärde sig AI:n att gömma kulan i ett
  hörn i stället för att våga sig förbi nästa hål.
- **Läroplan (curriculum):** `random_start=0.5` gör att hälften av omgångarna startar på
  en slumpvis punkt längs linjen, så AI:n får öva på slutet av banan tidigt.
- **Bilder:** `render_mode="rgb_array"` ger brädet som bild om vi senare vill träna på pixlar (CNN).

## Riktmärke

```powershell
.venv\Scripts\python -m labyrint.autopilot --episodes 20
```

Den handskrivna autopiloten klarar banan på ca 38 s. Den tränade AI:n (`models/labyrint_ai.zip`)
klarar den på ca 7 s: i stället för att följa linjen åker den längs väggarna och rundar hålen
på utsidan.

```powershell
.venv\Scripts\python -m labyrint.agent --episodes 100 --jitter 5   # mät AI:n: 99 % i mål
```

## Träna AI:n

```powershell
# en gång: torch (med CUDA) och stable-baselines3 m.fl.
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements-ai.txt

.venv\Scripts\python train.py --name ppo        # ca 35 min för 30 M steg -> runs/ppo/best_model.zip
```

### Se AI:n träna live

Dubbelklicka **`trana_ai_live.bat`**. Den startar en ny träning och öppnar två fönster:

- **Spelet** (`play.py --live`): AI:n spelar med sin *senaste* hjärna. Träningen sparar en
  ögonblicksbild var 3:e sekund och varje nytt försök använder den nyaste – man ser kulan gå
  från att ramla direkt till att klara hela banan (efter ca 2–3 minuter).
- **Graferna** (`dashboard.py`): andel i mål, tid till mål jämfört med autopiloten, belöning,
  förlust (value loss), hur mycket den fortfarande slumpar (utforskning) och hur väl den
  förutser sin belöning.

Själva träningen spelar 64 omgångar samtidigt, ca 500 gånger snabbare än realtid, så det man
ser i spelet är inte träningsomgångarna utan ett prov med den senaste versionen.
Stäng träningsfönstret (det svarta) för att avbryta; de andra två kan stängas och öppnas fritt:

```powershell
.venv\Scripts\python play.py --live          # följ den senaste körningen i runs/
.venv\Scripts\python dashboard.py
```

## Tester

```powershell
.venv\Scripts\python -m pytest
```

## Nästa steg

1. Välj bästa modell på prov med större startspridning – nu väljs den snabbaste, inte den stabilaste.
2. Fler och svårare banor; träna en AI som klarar banor den aldrig sett.
3. Träna på pixlar (`render_mode="rgb_array"`) i stället för sensorvärden.
