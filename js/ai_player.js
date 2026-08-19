/**
 * AI Player - Integrates the agents (DQN and Expectimax) with the 2048 game in the browser.
 * Connects to the Python server (port 8081) to get moves.
 * Saves a JSON report at the end of each game.
 */

(function () {
  var AI_SERVER = "http://localhost:8081";
  var aiPlaying = false;
  var aiSpeed = 0; // ms entre jogadas (0 = requestAnimationFrame)
  var aiAgent = "expectimax"; // "expectimax", "dqn" ou "ntuple"
  var agentList = ["expectimax", "dqn", "ntuple"];
  var agentLabels = {
    expectimax: "🧠 Expectimax",
    dqn: "🤖 DQN",
    ntuple: "🏆 N-Tuple"
  };
  var agentIndex = 0;
  var moveCount = 0;
  var moveHistory = [];  // move history for the report
  var gameStartTime = 0;

  // Extracts the grid as a 2D array [row][col] for the Python backend
  function extractGrid() {
    var gm = window.gameManager;
    if (!gm || !gm.grid) return null;

    var grid = [];
    for (var y = 0; y < gm.grid.size; y++) {
      var row = [];
      for (var x = 0; x < gm.grid.size; x++) {
        var cell = gm.grid.cells[x][y];
        row.push(cell ? cell.value : 0);
      }
      grid.push(row);
    }
    return grid;
  }

  // Requests the next move from the server
  function requestMove(grid, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", AI_SERVER + "/move", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4) {
        if (xhr.status === 200) {
          var data = JSON.parse(xhr.responseText);
          callback(null, data);
        } else {
          callback("AI server unavailable. Start: python ai/server.py");
        }
      }
    };
    xhr.onerror = function () {
      callback("AI server unavailable. Start: python ai/server.py");
    };
    xhr.send(JSON.stringify({ grid: grid, agent: aiAgent }));
  }

  // Sends report to the server
  function sendReport(won) {
    var gm = window.gameManager;
    if (!gm) return;

    var duration = Date.now() - gameStartTime;
    var report = {
      agent: aiAgent,
      score: gm.score,
      max_tile: getMaxTile(),
      moves: moveCount,
      won: won,
      final_grid: extractGrid(),
      move_history: moveHistory,
      duration_ms: duration
    };

    var xhr = new XMLHttpRequest();
    xhr.open("POST", AI_SERVER + "/report", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(report));
  }

  // Main AI loop
  function aiStep() {
    if (!aiPlaying) return;

    var gm = window.gameManager;
    if (!gm) return;

    if (gm.over || (gm.won && !gm.keepPlaying)) {
      aiPlaying = false;
      updateButton();
      var won = gm.won;
      var statusMsg = (won ? "WIN! " : "Game Over! ") +
        "Score: " + gm.score + " | Max: " + getMaxTile() + " | Moves: " + moveCount;
      updateStatus(statusMsg);
      sendReport(won);
      return;
    }

    var grid = extractGrid();
    if (!grid) return;

    requestMove(grid, function (err, data) {
      if (err) {
        aiPlaying = false;
        updateButton();
        updateStatus(err);
        return;
      }

      if (data.action >= 0) {
        moveCount++;
        var directions = ["↑", "→", "↓", "←"];
        var dirNames = ["up", "right", "down", "left"];
        var agentLabel = aiAgent.toUpperCase();

        // Record in history
        moveHistory.push({
          move: moveCount,
          action: dirNames[data.action],
          score_before: gm.score,
          max_tile: getMaxTile(),
          grid_before: grid
        });

        updateStatus(
          "[" + agentLabel + "] Move #" + moveCount + ": " + directions[data.action] +
          " | Score: " + gm.score + " | Max: " + getMaxTile()
        );

        // Dispatches the move via InputManager
        gm.inputManager.emit("move", data.action);
      }

      if (aiPlaying) {
        if (aiSpeed <= 0) {
          requestAnimationFrame(aiStep);
        } else {
          setTimeout(aiStep, aiSpeed);
        }
      }
    });
  }

  function getMaxTile() {
    var grid = extractGrid();
    if (!grid) return 0;
    var max = 0;
    for (var y = 0; y < grid.length; y++) {
      for (var x = 0; x < grid[y].length; x++) {
        if (grid[y][x] > max) max = grid[y][x];
      }
    }
    return max;
  }

  // UI
  function updateButton() {
    var btn = document.getElementById("ai-play-btn");
    if (btn) {
      btn.textContent = aiPlaying ? "⏹ Stop AI" : "🤖 AI Play";
      btn.className = "ai-button" + (aiPlaying ? " ai-active" : "");
    }
  }

  function updateStatus(msg) {
    var el = document.getElementById("ai-status");
    if (el) el.textContent = msg;
  }

  function toggleAI() {
    aiPlaying = !aiPlaying;
    updateButton();
    if (aiPlaying) {
      // If the game is over, restart
      var gm = window.gameManager;
      if (gm && (gm.over || (gm.won && !gm.keepPlaying))) {
        gm.inputManager.emit("restart");
      }
      moveCount = 0;
      moveHistory = [];
      gameStartTime = Date.now();
      var label = aiAgent.toUpperCase();
      updateStatus("[" + label + "] AI playing...");
      aiStep();
    } else {
      updateStatus("AI stopped.");
    }
  }

  function switchAgent() {
    agentIndex = (agentIndex + 1) % agentList.length;
    aiAgent = agentList[agentIndex];
    var btn = document.getElementById("ai-agent-btn");
    if (btn) {
      btn.textContent = agentLabels[aiAgent];
      var cls = "ai-agent-btn";
      if (aiAgent === "dqn") cls += " ai-agent-dqn";
      else if (aiAgent === "ntuple") cls += " ai-agent-ntuple";
      btn.className = cls;
    }
  }

  // Injects the controls into the page
  function injectUI() {
    var container = document.querySelector(".above-game");
    if (!container) return;

    var wrapper = document.createElement("div");
    wrapper.className = "ai-controls";
    wrapper.innerHTML =
      '<div class="ai-row">' +
      '  <a id="ai-play-btn" class="ai-button">🤖 AI Play</a>' +
      '  <a id="ai-agent-btn" class="ai-agent-btn">🧠 Expectimax</a>' +
      '</div>' +
      '<div id="ai-status" class="ai-status"></div>';

    container.parentNode.insertBefore(wrapper, container.nextSibling);

    document.getElementById("ai-play-btn").addEventListener("click", toggleAI);
    document.getElementById("ai-agent-btn").addEventListener("click", switchAgent);
  }

  // Injects CSS
  function injectStyles() {
    var style = document.createElement("style");
    style.textContent =
      ".ai-controls { text-align: center; margin: 10px 0; }" +
      ".ai-row { margin: 5px 0; }" +
      ".ai-button { display: inline-block; background: #8f7a66; color: #f9f6f2;" +
      "  border-radius: 3px; padding: 10px 20px; font-weight: bold; cursor: pointer;" +
      "  font-size: 16px; transition: background 0.15s; user-select: none; }" +
      ".ai-button:hover { background: #9f8b77; }" +
      ".ai-button.ai-active { background: #f65e3b; }" +
      ".ai-agent-btn { display: inline-block; background: #edc22e; color: #f9f6f2;" +
      "  border-radius: 3px; padding: 10px 16px; font-weight: bold; cursor: pointer;" +
      "  font-size: 14px; margin-left: 8px; transition: background 0.15s; user-select: none; }" +
      ".ai-agent-btn:hover { background: #f0d050; }" +
      ".ai-agent-btn.ai-agent-dqn { background: #e64c8a; }" +
      ".ai-agent-btn.ai-agent-dqn:hover { background: #f060a0; }" +
      ".ai-agent-btn.ai-agent-ntuple { background: #2ecc71; }" +
      ".ai-agent-btn.ai-agent-ntuple:hover { background: #40d87f; }" +
      ".ai-status { margin-top: 6px; font-size: 13px; color: #776e65;" +
      "  min-height: 20px; font-weight: bold; }";
    document.head.appendChild(style);
  }

  // Init
  window.addEventListener("DOMContentLoaded", function () {
    injectStyles();
    setTimeout(injectUI, 100);
  });
})();
