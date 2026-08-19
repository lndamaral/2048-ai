# Game Integrity Statement

## Original Game Code

The 2048 game engine is Gabriele Cirulli's original implementation, unmodified.
Source: https://github.com/gabrielecirulli/2048

### Files NOT modified (original game logic intact):

- `js/game_manager.js` — Game rules, move logic, merge, scoring, win/lose detection
- `js/grid.js` — Grid data structure
- `js/tile.js` — Tile data structure
- `js/html_actuator.js` — Rendering
- `js/keyboard_input_manager.js` — Input handling
- `js/local_storage_manager.js` — Save/load
- `style/` — All CSS unchanged

### Files modified (minimal, non-game-logic changes):

1. **`js/application.js`** — One line changed:
   ```diff
   - new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
   + window.gameManager = new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
   ```
   Purpose: Expose the game manager instance globally so the AI player can read the board state. Does NOT change game behavior.

2. **`index.html`** — One line added:
   ```html
   <script src="js/ai_player.js"></script>
   ```
   Purpose: Load the AI player script. Does NOT change game behavior.

### Files added (AI-only, do not affect game):

- `js/ai_player.js` — AI integration UI and server communication

### Game rules verification

The AI agents interact with the game exclusively through:
- Reading: `window.gameManager.grid.cells` (board state)
- Acting: `gm.inputManager.emit("move", direction)` (same as keyboard input)
- Continuing: `gm.keepPlaying = true` (same as clicking "Keep going")

No game rules, probabilities, or mechanics are altered. The AI receives the same game as a human player — same 90/10 probability for 2/4 tiles, same merge rules, same scoring.

### Reproducibility

To verify game integrity, compare our game files against the original:
```bash
# Fetch original
curl -s https://raw.githubusercontent.com/gabrielecirulli/2048/master/js/game_manager.js | md5
md5 js/game_manager.js
# These should match
```

### Python game engine

The Python game engine (`ai/game.py`) is an independent reimplementation used only for training and benchmarking. It faithfully reproduces the original JavaScript logic. The browser game uses the original JavaScript engine exclusively.
